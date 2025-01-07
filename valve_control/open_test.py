import serial
import time
import serial.tools.list_ports
import threading

# Find a connected Arduino
def find_arduino(description='Arduino'):
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        if description in port.description:
            return port.device
    return None

# Set up the serial connection
arduino_port = find_arduino('EDBG')
arduino_baudrate = 9600

if arduino_port:
    ser = serial.Serial(arduino_port, arduino_baudrate, timeout=1)
    print(f'Arduino found on port {arduino_port}, connecting...')
    time.sleep(2)  # Wait for the connection to establish
else:
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