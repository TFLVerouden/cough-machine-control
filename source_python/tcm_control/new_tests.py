import contextlib
import os
import time
from typing import Optional

from dvg_devices.BaseDevice import SerialDevice
from tcm_utils.file_dialogs import ask_open_file, find_repo_root, get_config_path


# PoF serial class that invokes the auto connection features on initialisation
class PoFSerialDevice(SerialDevice):
    def __init__(
        self,
        name: str,
        long_name: str,
        expected_id: str,
        baudrate: int = 115200,
        timeout: float = 1,
        debug: bool = False,
    ):
        super().__init__(name=name, long_name=long_name)
        self.serial_settings["baudrate"] = baudrate
        self.serial_settings["timeout"] = timeout

        last_known_port_path = get_config_path(f"{name}_path.txt")

        def id_query() -> tuple[str, None]:
            _success, reply = self.query("id?")
            if isinstance(reply, str):
                reply_broad = reply.strip()
            else:
                reply_broad = ""
            return reply_broad, None

        self.set_ID_validation_query(
            ID_validation_query=id_query,
            valid_ID_broad=expected_id,
        )

        # Auto connect to device; suppress print of connection attempts
        if debug:
            connected = self.auto_connect(
                filepath_last_known_port=str(last_known_port_path)
            )
        else:
            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                connected = self.auto_connect(
                    filepath_last_known_port=str(last_known_port_path)
                )

        if not connected:
            raise SystemError(
                f"Serial device {name} not found via auto_connect")
        else:
            print(f"Connected to serial device {name} at {self.ser.port}")

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------
    def _read_lines(self, timeout: float = 0.2) -> list[str]:
        # Drain any pending serial lines within the timeout window.
        lines: list[str] = []
        start = time.time()
        while (time.time() - start) < timeout:
            if self.ser is not None and self.ser.in_waiting > 0:
                success, line = self.readline()
                if success and isinstance(line, str):
                    lines.append(line)
            else:
                time.sleep(0.02)
        return lines

    def _check_errors(self, lines: list[str], raise_on_error: bool) -> bool:
        # Detect MCU error lines and optionally raise.
        for line in lines:
            if line.startswith("ERROR"):
                if raise_on_error:
                    raise RuntimeError(line)
                return False
        return True

    def _query_and_drain(
        self,
        cmd: str,
        expected: Optional[str] = None,
        expected_prefix: Optional[str] = None,
        raise_on_error: bool = True,
        echo: bool = True,
        extra_timeout: float = 0.2,
    ) -> tuple[Optional[str], list[str]]:
        # Issue a query, collect additional lines, and validate responses.
        success, reply = self.query(cmd, raises_on_timeout=True)
        if not success:
            raise RuntimeError(f"Query failed: {cmd}")

        lines: list[str] = []
        if isinstance(reply, str):
            lines.append(reply)
        lines.extend(self._read_lines(timeout=extra_timeout))

        if echo:
            for line in lines:
                print(line)

        self._check_errors(lines, raise_on_error=raise_on_error)

        if expected is not None and reply != expected:
            raise RuntimeError(
                f"Unexpected reply to {cmd}: {reply!r} (expected {expected!r})"
            )
        if expected_prefix is not None and (
            not isinstance(reply, str) or not reply.startswith(expected_prefix)
        ):
            raise RuntimeError(
                f"Unexpected reply to {cmd}: {reply!r} (expected prefix {expected_prefix!r})"
            )

        return reply if isinstance(reply, str) else None, lines


class CoughMachine(PoFSerialDevice):
    def __init__(
        self,
        name: str = "CoughMachine_MCU",
        long_name: str = "Adafruit ItsyBitsy M4 Express",
        expected_id: str = "TCM_control",
        baudrate: int = 115200,
        timeout: float = 1,
        debug: bool = False,
    ):
        super().__init__(
            name=name,
            long_name=long_name,
            expected_id=expected_id,
            baudrate=baudrate,
            timeout=timeout,
            debug=debug
        )

        self._wait_us: Optional[int] = None
        self._dataset_loaded = False

        # Set debug mode on device if requested
        self.set_debug(debug)

    # ------------------------------------------------------------------
    # Serial command wrappers
    # ------------------------------------------------------------------
    def set_debug(self, enabled: bool, *, echo: bool = True) -> None:
        cmd = "B 1" if enabled else "B 0"
        expected = "DEBUG_ON" if enabled else "DEBUG_OFF"
        self._query_and_drain(cmd, expected=expected, echo=echo)
        print(f"Debug mode {'enabled' if enabled else 'disabled'} on device.")

    def set_valve_current(self, current_ma: float, *, echo: bool = True) -> str:
        reply, _lines = self._query_and_drain(
            f"V {current_ma}", expected_prefix="SET_VALVE", echo=echo
        )
        return reply or ""

    def set_pressure(self, pressure_bar: float, *, echo: bool = True) -> str:
        reply, _lines = self._query_and_drain(
            f"P {pressure_bar}", expected_prefix="SET_PRESSURE", echo=echo
        )
        return reply or ""

    def read_pressure(self, *, echo: bool = True) -> Optional[float]:
        reply, _lines = self._query_and_drain(
            "P?", expected_prefix="P", echo=echo)
        if reply is None:
            return None
        try:
            return float(reply[1:])
        except ValueError:
            return None

    def set_wait_us(self, wait_us: int, *, echo: bool = True) -> str:
        reply, _lines = self._query_and_drain(
            f"W {wait_us}", expected_prefix="SET_WAIT", echo=echo
        )
        self._wait_us = wait_us
        return reply or ""

    def read_wait_us(self, *, echo: bool = True) -> Optional[int]:
        reply, _lines = self._query_and_drain(
            "W?", expected_prefix="W", echo=echo)
        if reply is None:
            return None
        try:
            return int(reply[1:])
        except ValueError:
            return None

    def dataset_status(self, *, echo: bool = True) -> str:
        reply, _lines = self._query_and_drain("L?", echo=echo)
        return reply or ""

    def open_solenoid(self, *, echo: bool = True) -> str:
        reply, _lines = self._query_and_drain(
            "O", expected="SOLENOID_OPENED", echo=echo)
        return reply or ""

    def close_solenoid(self, *, echo: bool = True) -> str:
        reply, _lines = self._query_and_drain(
            "C", expected="SOLENOID_CLOSED", echo=echo)
        return reply or ""

    def clear_memory(self, *, echo: bool = True) -> str:
        reply, _lines = self._query_and_drain(
            "C!", expected="MEMORY_CLEARED", echo=echo)
        return reply or ""

    def droplet_detect(self, runs: Optional[int] = None, *, echo: bool = True) -> str:
        cmd = "D" if runs is None else f"D {runs}"
        reply, _lines = self._query_and_drain(
            cmd, expected="DROPLET_ARMED", echo=echo)
        return reply or ""

    def laser_test(self, enabled: bool, *, echo: bool = True) -> str:
        cmd = "A 1" if enabled else "A 0"
        expected = "LASER_TEST_ON" if enabled else "LASER_TEST_OFF"
        reply, _lines = self._query_and_drain(
            cmd, expected=expected, echo=echo)
        return reply or ""

    def read_temperature_humidity(self, *, echo: bool = True) -> tuple[Optional[float], Optional[float]]:
        reply, _lines = self._query_and_drain(
            "T?", expected_prefix="T", echo=echo)
        if reply is None:
            return None, None
        try:
            parts = reply.split()
            temp = float(parts[0][1:])
            hum = float(parts[1][1:])
            return temp, hum
        except (IndexError, ValueError):
            return None, None

    def read_status(self, *, echo: bool = True, timeout: float = 1.0) -> list[str]:
        if not self.write("S?"):
            raise RuntimeError("Failed to send S? command")

        lines = self._read_lines(timeout=timeout)
        if echo:
            for line in lines:
                print(line)
        self._check_errors(lines, raise_on_error=True)
        return lines

    def identify(self, *, echo: bool = True) -> str:
        reply, _lines = self._query_and_drain("id?", echo=echo)
        return reply or ""

    def help(self, *, echo: bool = True) -> str:
        reply, _lines = self._query_and_drain("?", echo=echo)
        return reply or ""

    def load_dataset(self) -> None:
        raise NotImplementedError("Dataset upload not implemented yet.")

    def run(self) -> None:
        raise NotImplementedError("Run command needs a multi-line handler.")


# Class variables for testing
device = CoughMachine(debug=True)
