import propar
import serial
import time
import serial.tools.list_ports
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
print(sht.temperature)
print(sht.relative_humidity)
import adafruit_sht4x

sht = adafruit_sht4x.SHT4x(board.I2C())
print(sht.temperature)
print(sht.relative_humidity)
# Find a connected serial device by description
def find_serial_device(description):
    ports = list(serial.tools.list_ports.comports())
    ports.sort(key=lambda port: int(port.device.replace('COM', '')))

    available_ports = [port.device for port in ports
                       if description in port.description]

    if len(available_ports) == 1:
        return available_ports[0]
    else:
        print('Available devices:')
        for port in ports:
            print(f'{port.device} - {port.description}')

        choice = input(f'Enter the COM port number for "{description}": COM')
        return f'COM{choice}'


arduino_port = find_serial_device(description='Adafruit')
arduino_baudrate = 115200

if arduino_port:
    ser = serial.Serial(arduino_port, arduino_baudrate, timeout=0)  # Non-blocking mode
    time.sleep(0.2)  # Slightly reduced connection wait time
    print(f'Connected to Arduino on {arduino_port}')
else:
    raise SystemError('Arduino not found')

# Set up the flow meter serial connection
flow_meter_port = find_serial_device(description='Bronkhorst')
flow_meter_baudrate = 115200  # 38400
flow_meter_node = 3

if flow_meter_port:
    flow_meter = propar.instrument(comport=flow_meter_port,
                                   address=flow_meter_node,
                                   baudrate=flow_meter_baudrate)
    flow_meter_serial_number = flow_meter.readParameter(92)
    time.sleep(1)  # Reduced connection wait time
    print(f'Connected to flow meter {flow_meter_serial_number} on {flow_meter_port}')
else:
    raise SystemError('Flow meter not found')

# Send the command to enable continuous streaming (this is an example, check the flow meter's documentation)
# Some devices have a specific command to start continuous data output, e.g., b'CONTINUOUS\r'

start_time = time.time()
duration_s = 2 # Set the desired duration of continuous reading (in seconds)

# Read the data stream continuously
while (time.time() - start_time) < duration_s:
    # Non-blocking read from the serial buffer

    time_before = time.time()
    flow_meter_value = float(flow_meter.readParameter(8))  # Adjust based on the continuous data format
    time_after = time.time()
    print(f"Flow meter value: {flow_meter_value},time: { (time_after - time_before)*1E-6}")

# Stop continuous streaming (if applicable)

# Close the serial port connection
ser.close()
