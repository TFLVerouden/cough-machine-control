import serial
import time
import serial.tools.list_ports

def find_serial_device(description, continue_on_error=False):
    ports = list(serial.tools.list_ports.comports())
    ports.sort(key=lambda port: int(port.device.replace('COM', '')))

    # Filter ports where the description contains the provided keyword
    matching_ports = [port.device for port in ports if description in port.description]

    if len(matching_ports) == 1:
        return matching_ports[0]
    elif len(matching_ports) > 1:
        print('Multiple matching devices found:')
        for idx, port in enumerate(ports):
            print(f'{idx+1}. {port.device} - {port.description}')
        choice = input(f'Select the device number for "{description}": ')
        return matching_ports[int(choice) - 1]
    else:
        if continue_on_error:
            return None
        print('No matching devices found. Available devices:')
        for port in ports:
            print(f'{port.device} - {port.description}')
        choice = input(f'Enter the COM port number for "{description}": COM')
        return f'COM{choice}'

class SprayTecLift(serial.Serial):
    def __init__(self, port, baudrate=9600, timeout=1):
        super().__init__(port=port, baudrate=baudrate, timeout=timeout)
        time.sleep(1)  # Allow time for the connection to establish
        print(f"Connected to SprayTec lift on {port}")

    def get_height(self):
        """Send a command to get the platform height and parse the response."""
        try:
            self.write(b'?\n')  # Send the status command
            response = self.readlines()
            for line in response:
                if line.startswith(b'  Platform height [mm]: '):
                    height = line.split(b': ')[1].strip().decode('utf-8')
                    return float(height)
            print('Warning: No valid response containing "Platform height [mm]" was found.')
            return None
        except Exception as e:
            print(f"Error while reading lift height: {e}")
            return None

    def close_connection(self):
        """Close the serial connection."""
        self.close()
        # print("Lift connection closed.")


if __name__ == '__main__':
    # Find the SprayTec lift port
    lift_port = find_serial_device(description='Mega', continue_on_error=True)

    if lift_port:
        lift = SprayTecLift(port=lift_port)
        height = lift.get_height()
        if height is not None:
            print(f"Lift height: {height} mm")
        lift.close_connection()
    else:
        print("Warning: Unable to find the SprayTec lift; height not recorded.")