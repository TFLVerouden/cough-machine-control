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
        verbose: bool = False,
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
        if verbose:
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

    def receive_data(self, str_start, str_end, timeout_sec=10)


# Class variables for testing
NAME = "TCM_control"
LONG_NAME = "Adafruit ItsyBitsy M4 Feather Express"
ID = "TCM_control"
BAUD_RATE = 115200
TIMEOUT = 1
VERBOSE = False

last_known_port_path = get_config_path("TCM_control_path.txt")

device = PoFSerialDevice(
    name=NAME,
    long_name=LONG_NAME,
    expected_id=ID,
    baudrate=BAUD_RATE,
    timeout=TIMEOUT,
    verbose=VERBOSE
)
