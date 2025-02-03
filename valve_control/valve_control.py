import propar
import serial
import time
import serial.tools.list_ports
import datetime
import csv
import os
import numpy as np

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


if __name__ == '__main__':
    # Create the data directory if it doesn't exist
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Get the experiment name
    experiment_name_default = "test"
    experiment_name = (input(f'Enter experiment name (press ENTER for '
                             f'"{experiment_name_default}"): ').strip()
                       or experiment_name_default)

    # Get the duration of the valve opening
    duration_ms_default = 500
    duration_ms = int(input(f'Enter valve opening duration (press ENTER for '
                            f'{duration_ms_default} ms): ').strip()
                      or duration_ms_default)

    # Set the before and after times
    before_time_ms = 500
    after_time_ms = 1000

    # Set up the Arduino serial connection
    arduino_port = find_serial_device(description='Adafruit')
    arduino_baudrate = 115200

    if arduino_port:
        ser = serial.Serial(arduino_port, arduino_baudrate,
                            timeout=0)  # Non-blocking mode
        time.sleep(1)  # Wait for the connection to establish
        print(f'Connected to Arduino on {arduino_port}')
    else:
        raise SystemError('Arduino not found')

    # Set up the flow meter serial connection
    flow_meter_port = find_serial_device(description='Bronkhorst')
    flow_meter_baudrate = 38400
    flow_meter_node = 3

    if flow_meter_port:
        flow_meter = propar.instrument(comport=flow_meter_port,
                                       address=flow_meter_node,
                                       baudrate=flow_meter_baudrate)
        flow_meter_serial_number = flow_meter.readParameter(92)
        time.sleep(1)  # Wait for the connection to establish
        print(
                f'Connected to flow meter {flow_meter_serial_number} '
                f'on {flow_meter_port}')
    else:
        raise SystemError('Flow meter not found')

    # Record the start time
    start_time = datetime.datetime.now(datetime.timezone.utc)
    readings = np.array([],dtype=float)

    # Start the while loop
    valve_opened = False
    finished_received = False
    loop_start_time = time.time()

    print('Starting experiment...')

    while True:
        # TODO: Speed up while loop. Problem seems to be waiting for
        #  an Arduino response
        current_time = time.time()
        elapsed_time = current_time - loop_start_time

        # Listen for commands from the Arduino
        if ser.in_waiting > 0:
            response = ser.readline().decode('utf-8').rstrip()
            if response == "!":
                print('Valve closed')
                finished_received = True
                finished_time = current_time
            else:
                # Assume it's a pressure value
                pressure_value = float(response)

                # Read the flow meter value
                flow_meter_value = float(flow_meter.readParameter(8))

                readings = np.append(readings, [elapsed_time, pressure_value, flow_meter_value])


        # Ask the Arduino for a single pressure readout
        ser.write('P?\n'.encode())

        # After a set time, send a command to the Arduino to open the valve
        if not valve_opened and elapsed_time >= (before_time_ms / 1000):
            print('Opening valve...')
            ser.write(f'O {duration_ms}\n'.encode())
            valve_opened = True

        # Continue the loop for an additional time after receiving "!"
        if finished_received and (current_time - finished_time) >= (
                after_time_ms / 1000):
            # Todo: If, for whatever reason, no "!" is received, the loop
            #  will continue indefinitely
            print('Experiment finished')
            break

    # Save the readings to a CSV file
    print('Saving data...')
    flow_meter_calibration_value = 10 / 30000 #L/s at maximum capacity: 30.000 a.u.
    readings = readings.reshape(-1,3)
    readings[:,2] = readings[:,2] * flow_meter_calibration_value  #now in L/s
    timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M')
    filename = os.path.join(data_dir, f'{timestamp}_{experiment_name}.csv')
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header
        csvwriter.writerow(['Experiment name', experiment_name])
        csvwriter.writerow(['Start time (UTC)', start_time])
        csvwriter.writerow(['Opening duration (ms)', duration_ms])
        csvwriter.writerow(['Time before opening (ms)', before_time_ms])
        csvwriter.writerow(['Time after closing (ms)', after_time_ms])
        csvwriter.writerow([])
        csvwriter.writerow(
                ['Elapsed time (s)', 'Pressure (bar)', 'Flow rate (L/s)'])

        # Write the readings
        for reading in readings:
            csvwriter.writerow(reading)
