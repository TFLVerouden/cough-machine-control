import serial
import time
import serial.tools.list_ports
import threading

# Find a connected Arduino
def find_arduino(description='Arduino'):
    ports = list(serial.tools.list_ports.comports())
    ports.sort(key=lambda port: int(port.device.replace('COM', '')))

    available_ports = [port.device for port in ports if description in port.description]

    if len(available_ports) == 1:
        return available_ports[0]
    else:
        print('Available devices:')
        for port in ports:
            print(f'{port.device} - {port.description}')

        choice = input(f'Enter the {description} COM port number (e.g., 3 for COM3): ')
        return f'COM{choice}'


# Set up the serial connection
arduino_port = find_arduino()
arduino_baudrate = 9600

if arduino_port:
    ser = serial.Serial(arduino_port, arduino_baudrate, timeout=1)
    print(f'Arduino found on port {arduino_port}, connecting...')
    time.sleep(2)  # Wait for the connection to establish
else:
    # Print a list of all the device descriptions
    print('No Arduino found, available devices:')
    for port in list(serial.tools.list_ports.comports()):
        print(port.description)
    # Stop the script if the Arduino is not found
    raise SystemError('Arduino not found')

def set_time(duration_ms):
    ser.write(f'TIME {duration_ms}\n'.encode())

def open_valve():
    ser.write('OPEN\n'.encode())

def reset_time():
    ser.write('RESET\n'.encode())

def monitor_serial():
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').rstrip()
            print(f'\r{line}\nEnter command: ', end='')

def manual_input():
    while True:
        command = input("Enter command: ")
        ser.write(f'{command}\n'.encode())

# Start the serial monitor in a separate thread
serial_thread = threading.Thread(target=monitor_serial)
serial_thread.daemon = True
serial_thread.start()

# Start the manual input loop
manual_input()

# Test the functions
# set_time(5000)
# open_valve()
# reset_time()