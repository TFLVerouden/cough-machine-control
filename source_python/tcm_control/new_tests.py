import contextlib
import os

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

        if debug:
            if self.query("B 1", raises_on_timeout=True)[1] == "DEBUG_ON":
                print("Debug mode enabled on device.")
        else:
            if self.query("B 1")[1] == "DEBUG_OFF":
                print("Debug mode disabled on device.")


# Class variables for testing
device = CoughMachine(debug=True)
