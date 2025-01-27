import serial
import serial.tools.list_ports
import time
import threading

# Find a connected serial device by description
def find_serial_device(description=''):
    ports = list(serial.tools.list_ports.comports())
    ports.sort(key=lambda port: int(port.device.replace('COM', '')))

    available_ports = [port.device for port in ports if description in port.description]

    if len(available_ports) == 1:
        return available_ports[0]
    else:
        print('Available devices:')
        for port in ports:
            print(f'- {port.description}')

        choice = input(f'Enter the {description} port number: COM')
        return f'COM{choice}'


# Set up the Arduino connection
arduino_port = find_serial_device(description='Adafruit')
arduino_baudrate = 9600

ser_arduino = serial.Serial(arduino_port, arduino_baudrate, timeout=1)
print(f'Arduino found on port {arduino_port}, connecting...')
time.sleep(2)  # Wait for the connection to establish

# Set up flow meter connection
# flow_meter_port = find_serial_device()
# flow_meter_baudrate = 38400
# ser_flow_meter = serial.Serial(flow_meter_port, flow_meter_baudrate, timeout=1)
# print(f'Flow meter found on port {flow_meter_port}, connecting...')
# time.sleep(2)  # Wait for the connection to establish

def monitor_serial():
    while True:
        if ser_arduino.in_waiting > 0:
            line = ser_arduino.readline().decode('utf-8').rstrip()
            print(f'\r{line}\nEnter command: ', end='')

def manual_input():
    while True:
        command = input("Enter command: ")
        ser_arduino.write(f'{command}\n'.encode())

# Start the serial monitor in a separate thread
serial_thread = threading.Thread(target=monitor_serial)
serial_thread.daemon = True
serial_thread.start()

# # Start the manual input loop
manual_input()

# Test the functions
# ser_arduino.write('TIME 500\n'.encode())
# ser_arduino.write('OPEN\n'.encode())