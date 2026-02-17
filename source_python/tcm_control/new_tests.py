import contextlib
import csv
import os
import time
from pathlib import Path
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
        self._debug = debug
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
            # Drain any boot/session leftovers before issuing new commands.
            pending = self._read_lines(timeout=0.5)
            if pending:
                if self._debug:
                    for line in pending:
                        print(f"[{self.name}] {line}")
                self._check_errors(pending, raise_on_error=True)
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
                if raise_on_error and not self._debug:
                    raise RuntimeError(line)
                return False
        return True

    def _query_and_drain(
        self,
        cmd: Optional[str],
        expected: Optional[str] = None,
        expected_prefix: Optional[str] = None,
        raise_on_error: bool = True,
        echo: bool = True,
        extra_timeout: float = 0.2,
    ) -> tuple[Optional[str], list[str]]:
        # Issue a query (optional), collect additional lines, and validate responses.
        reply: Optional[str] = None
        lines: list[str] = []
        if cmd:
            success, reply = self.query(cmd, raises_on_timeout=True)
            if not success:
                raise RuntimeError(f"Query failed: {cmd}")
            if isinstance(reply, str):
                lines.append(reply)
        lines.extend(self._read_lines(timeout=extra_timeout))

        if echo:
            for line in lines:
                print(f"[{self.name}] {line}")

        self._check_errors(lines, raise_on_error=raise_on_error)

        if not self._debug:
            matched: Optional[str] = None
            if expected is not None:
                for line in lines:
                    if line == expected:
                        matched = line
                        break
            elif expected_prefix is not None:
                for line in lines:
                    if line.startswith(expected_prefix):
                        matched = line
                        break

            if matched is not None:
                reply = matched

            if expected is not None and reply != expected:
                raise RuntimeError(
                    f"Unexpected reply to {cmd}: {reply!r} (expected {expected!r})"
                )
            if expected_prefix is not None and (
                not isinstance(reply, str) or not reply.startswith(
                    expected_prefix)
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
        self._dataset_csv_path: Optional[Path] = None

        # Set debug mode on device if requested
        self._set_debug(debug)

    # ------------------------------------------------------------------
    # Manual mode
    # ------------------------------------------------------------------

    # Allow user to type commands directly to the device
    def manual_mode(self) -> None:
        print("Entering manual mode. Type commands to send to the device. Ctrl+C to exit.")
        try:
            while True:
                cmd = input(">> ")
                if cmd.strip().lower() in {"exit", "quit"}:
                    print("Exiting manual mode.")
                    break
                self._query_and_drain(cmd, echo=True, raise_on_error=False)
        except KeyboardInterrupt:
            print("\nExiting manual mode.")

    # ------------------------------------------------------------------
    # Serial command wrappers
    # ------------------------------------------------------------------

    # CONNECTION & DEBUGGING
    def _identify(self, *, echo: bool = True) -> str:
        reply, _lines = self._query_and_drain("id?", echo=echo)
        return reply or ""

    def _set_debug(self, enabled: bool) -> None:
        cmd = "B 1" if enabled else "B 0"
        expected = "DEBUG_ON" if enabled else "DEBUG_OFF"
        self._query_and_drain(cmd, expected=expected, echo=enabled)
        if enabled:
            print("Debug mode enabled on device.")

    def read_status(self, *, echo: bool = True, timeout: float = 1.0) -> list[str]:
        if not self._debug:
            raise RuntimeError("read_status is only available in debug mode.")
        if not self.write("S?"):
            raise RuntimeError("Failed to send S? command")

        lines = self._read_lines(timeout=timeout)
        if echo:
            for line in lines:
                print(f"[{self.name}] {line}")
        self._check_errors(lines, raise_on_error=True)
        return lines

    def help(self, *, echo: bool = True) -> str:
        reply, _lines = self._query_and_drain("?", echo=echo)
        return reply or ""

    # CONTROL HARDWARE
    def set_valve_current(self, current_ma: float, *, echo: bool = True) -> str:
        reply, _lines = self._query_and_drain(
            f"V {current_ma}", expected_prefix="SET_VALVE", echo=echo
        )
        return reply or ""

    def set_pressure(
        self,
        pressure_bar: float,
        *,
        timeout_s: float = 60.0,
        avg_window_s: float = 5.0,
        tolerance_bar: float = 0.05,
        poll_interval_s: float = 0.2,
        echo: bool = True,
    ) -> str:
        reply, _lines = self._query_and_drain(
            f"P {pressure_bar}", expected_prefix="SET_PRESSURE", echo=echo
        )

        # Check parameters
        if timeout_s <= 0:
            return reply or ""
        if avg_window_s <= 0:
            raise ValueError("avg_window_s must be > 0")
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be > 0")

        # Loop until we reach the setpoint within tolerance,
        # using a rolling average to smooth out noise
        start = time.time()
        samples: list[tuple[float, float]] = []
        first_sample_time = time.time()
        while (time.time() - start) < timeout_s:
            reading = self.read_pressure(echo=False)
            if reading is not None:
                now = time.time()
                samples.append((now, reading))
                cutoff = now - avg_window_s
                samples = [(t, p) for t, p in samples if t >= cutoff]

                deviation = reading - pressure_bar
                print(
                    f"\rPressure: {reading:.2f} bar (dev {deviation:+.2f})",
                    end="",
                    flush=True,
                )

                if (now - first_sample_time) >= avg_window_s and samples:
                    avg = sum(p for _, p in samples) / len(samples)
                    if abs(avg - pressure_bar) <= tolerance_bar:
                        print()
                        return reply or ""
            else:
                print("\rPressure: -.-- bar (dev ---)", end="", flush=True)

            time.sleep(poll_interval_s)

        print()
        raise RuntimeError(
            "Could not reach setpoint value or pressure too unstable."
        )

    def open_solenoid(self, *, echo: bool = True) -> str:
        reply, _lines = self._query_and_drain(
            "O", expected="SOLENOID_OPENED", echo=echo)
        return reply or ""

    def close_solenoid(self, *, echo: bool = True) -> str:
        reply, _lines = self._query_and_drain(
            "C", expected="SOLENOID_CLOSED", echo=echo)
        return reply or ""

    def laser_test(
        self,
        enabled: bool = True,
        *,
        duration_s: Optional[float] = None,
        echo: bool = True,
    ) -> str:
        if duration_s is not None and enabled:
            reply_on, _lines_on = self._query_and_drain(
                "A 1", expected="LASER_TEST_ON", echo=echo
            )
            time.sleep(duration_s)
            reply_off, _lines_off = self._query_and_drain(
                "A 0", expected="LASER_TEST_OFF", echo=echo
            )
            return reply_off or reply_on or ""

        cmd = "A 1" if enabled else "A 0"
        expected = "LASER_TEST_ON" if enabled else "LASER_TEST_OFF"
        reply, _lines = self._query_and_drain(
            cmd, expected=expected, echo=echo)
        return reply or ""

    # READ OUT SENSORS
    def read_pressure(self, *, echo: bool = True) -> Optional[float]:
        reply, _lines = self._query_and_drain(
            "P?", expected_prefix="P", echo=echo)
        if reply is None:
            return None
        try:
            return float(reply[1:])
        except ValueError:
            return None

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

    # CONFIGURATION
    def set_wait_us(self, wait_us: int, *, echo: bool = True) -> str:
        reply, _lines = self._query_and_drain(
            f"W {wait_us}", expected_prefix="SET_WAIT", echo=echo
        )
        self._wait_us = wait_us
        return reply or ""

    def get_wait_us(self, *, echo: bool = True) -> Optional[int]:
        reply, _lines = self._query_and_drain(
            "W?", expected_prefix="W", echo=echo)
        if reply is None:
            return None
        try:
            return int(reply[1:])
        except ValueError:
            return None

    def clear_memory(self, *, echo: bool = True) -> str:
        reply, _lines = self._query_and_drain(
            "C!", expected="MEMORY_CLEARED", echo=echo)
        return reply or ""

    # DATASET HANDLING
    def set_dataset_csv_path(self, csv_path: str | Path | None) -> None:
        # Store a default path for later; load_dataset() will use this if no path is passed.
        self._dataset_csv_path = Path(
            csv_path) if csv_path is not None else None

    def load_dataset(
        self,
        csv_path: str | Path | None = None,
        *,
        delimiter: str = ",",
        echo: bool = True,
        timeout: float = 1.0,
    ) -> str:
        # If a path is passed here, it overrides any previously stored default.
        if csv_path is not None:
            self._dataset_csv_path = Path(csv_path)

        # If no path was provided or stored, fall back to the file picker dialog.
        if self._dataset_csv_path is None:
            repo_root = find_repo_root()
            self._dataset_csv_path = ask_open_file(
                key="flow_curve_csv",
                title="Select flow curve CSV",
                filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
                default_dir=repo_root / "source_python" / "flow_curves",
                start=repo_root,
            )

        if self._dataset_csv_path is None:
            raise SystemExit("No flow curve CSV selected")

        time_arr, mA_arr, enable_arr = self._extract_csv(
            self._dataset_csv_path, delimiter=delimiter)
        serial_command = self._format_dataset(time_arr, mA_arr, enable_arr)

        if self._debug:
            print(f"Formatted serial command:\n{serial_command}")

        if not self.write(serial_command):
            raise RuntimeError("Failed to write dataset to device.")

        # Wait for dataset upload confirmation without issuing extra commands.
        reply, _lines = self._query_and_drain(
            None,
            expected="DATASET_SAVED",
            echo=echo,
            extra_timeout=timeout,
        )

        self._dataset_loaded = True
        return reply or ""

    def get_dataset_status(self, *, echo: bool = True) -> str:
        reply, _lines = self._query_and_drain("L?", echo=echo)
        return reply or ""

    # COUGH
    def run(self) -> None:
        raise NotImplementedError("Run command needs a multi-line handler.")

    def detect_droplet(self, runs: Optional[int] = None, *, echo: bool = True) -> str:
        cmd = "D" if runs is None else f"D {runs}"
        reply, _lines = self._query_and_drain(
            cmd, expected="DROPLET_ARMED", echo=echo)
        return reply or ""

    # -------------------------------------------------------------------
    # Dataset read and upload
    # -------------------------------------------------------------------

    @staticmethod
    def _extract_csv(filename: str | Path, delimiter: str = ",") -> tuple[list[str], list[str], list[str]]:
        # Parse a CSV file into time, current, enable arrays for the L command.
        time_arr: list[str] = []
        mA_arr: list[str] = []
        enable_arr: list[str] = []
        row_idx = 0

        with open(filename, "r") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=delimiter)
            for rows in csvreader:
                if len(rows) < 3 or not rows[0] or not rows[1] or rows[2] == "":
                    raise ValueError(
                        f"Encountered empty cell at row index {row_idx}!")
                # replace ',' with '.' depending on csv format (';' delim vs ',' delim)
                time_arr.append(rows[0].replace(",", "."))
                mA_arr.append(rows[1].replace(",", "."))
                enable_arr.append(rows[2].strip())
                row_idx += 1

        if not time_arr or not mA_arr or not enable_arr:
            raise ValueError("CSV contains no data.")
        return time_arr, mA_arr, enable_arr

    @staticmethod
    def _format_dataset(
        time_array: list[str],
        mA_array: list[str],
        enable_array: list[str],
        *,
        prefix: str = "L",
        handshake_delim: str = " ",
        data_delim: str = ",",
    ) -> str:
        # Format the arrays into the serial protocol for dataset upload.
        if (
            not time_array
            or len(time_array) != len(mA_array)
            or len(time_array) != len(enable_array)
        ):
            raise ValueError(
                f"Arrays are not compatible! Time length: {len(time_array)}, "
                f"mA length: {len(mA_array)}, enable length: {len(enable_array)}"
            )

        duration = time_array[-1]
        header = (
            f"{prefix}{handshake_delim}{len(time_array)}"
            f"{handshake_delim}{duration}{handshake_delim}"
        )
        data = [
            str(val)
            for t, mA, e in zip(time_array, mA_array, enable_array)
            for val in (t, mA, e)
        ]
        return header + data_delim.join(data)

    # -------------------------------------------------------------------
    # Run logging
    # -------------------------------------------------------------------


if __name__ == "__main__":

    cough_machine = CoughMachine(debug=False)
    cough_machine.load_dataset(
        csv_path="C:\\Users\\local2\\Documents\\GitHub\\cough-machine-control\\source_python\\tcm_control\\flow_curves\\default_curve_new.csv")
    cough_machine.manual_mode()
