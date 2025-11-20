import pumpy3
import serial
import time
import serial.tools.list_ports

# FUNCTIONS THAT NEED TO BE MOVED ELSEWHERE AT SOME POINT


def find_serial_device(description, continue_on_error=False):
    ports = list(serial.tools.list_ports.comports())
    ports.sort(key=lambda port: int(port.device.replace('COM', '')))

    # Filter ports where the description contains the provided keyword
    matching_ports = [
        port.device for port in ports if description in port.description]

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


def read_temperature(verbose=False):
    ser_mcu.write('T?\n'.encode())
    time.sleep(0.1)  # wait for the response
    Temperature = ser_mcu.readline().decode('utf-8').rstrip()
    RH = ser_mcu.readline().decode('utf-8').rstrip()
    Temperature = Temperature.lstrip('T')
    RH = RH.lstrip('RH')

    if verbose:
        print(f'Temperature: {Temperature} °C; relative humidity: {RH} %')
    return RH, Temperature


# INITIALISE VALVE CONTROLLER
mcu_port = find_serial_device('ItsyBitsy')
mcu_baudrate = 115200

if mcu_port is None:
    raise Exception(
        "MCU port not found. Please connect the device and try again.")

ser_mcu = serial.Serial(mcu_port, mcu_baudrate, timeout=0)
time.sleep(1)
print(f"Connected to MCU on {mcu_port} at {mcu_baudrate} baud.")


# INITIALISE PUMP
# Initialise chain and PHD 2000
chain = pumpy3.Chain(
    "COM10",        # Manually specified, no way to auto-detect (yet)
    baudrate=19200,  # Set to match pump
    timeout=0.3     # 300 ms timeout, increase if unstable
)
pump = pumpy3.PumpPHD2000_Refill(chain, address=0, name="PHD2000")

# Configure pump
pump.set_diameter(10.3)     # 5 mL Hamilton gastight nr 1005
pump.set_mode("PMP")        # Set to PuMP mode
pump.set_rate(0.2, "ml/mn")  # Flow rate

input("Press Enter to start flushing the pump...")

# Set MCU delay to 0 us
ser_mcu.write('L 0\n'.encode())

# Flush pump for a bit
pump.run()
time.sleep(2)

# Set MCU to droplet detection mode without opening valve
ser_mcu.write('D 1\n'.encode())

# When first droplet is detected, stop pump
if ser_mcu.readline().decode('utf-8').rstrip() == '!':
    pump.stop()
