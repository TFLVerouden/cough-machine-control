import serial
import time
import serial.tools.list_ports

# Find a connected Arduino
def find_arduino():
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        if 'Arduino' in port.description or 'EDBG' in port.description:
            return port.device
    return None

# Set up the serial connection (adjust the port and baud rate as needed)
arduino_port = find_arduino()
if arduino_port:
    ser = serial.Serial(arduino_port, 9600, timeout=1)
    print(f'Arduino found on port {arduino_port}, connecting...')
    time.sleep(2)  # Wait for the connection to establish
else:
    # Stop the script if the Arduino is not found
    raise SystemError('Arduino not found')

def set_time(duration_ms):
    ser.write(f'TIME {duration_ms}\n'.encode())

def open_valve():
    ser.write('OPEN\n'.encode())

def monitor_serial():
    print('Serial monitor:')
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').rstrip()
            print(line)

# Example usage
set_time(5000)  # Set the timing duration to 5000 ms (5 seconds)
time.sleep(2)  # Wait for 2 seconds
open_valve()  # Open the valve

# Continuously read from Arduino
monitor_serial()