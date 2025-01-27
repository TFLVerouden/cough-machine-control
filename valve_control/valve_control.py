import serial
import time
import serial.tools.list_ports
import datetime
import csv

# Find a connected serial device by description
def find_serial_device(description):
    ports = list(serial.tools.list_ports.comports())
    ports.sort(key=lambda port: int(port.device.replace('COM', '')))

    available_ports = [port.device for port in ports if description in port.description]

    if len(available_ports) == 1:
        return available_ports[0]
    else:
        print('Available devices:')
        for port in ports:
            print(f'{port.device} - {port.description}')

        choice = input(f'Enter the COM port number for "{description}": COM')
        return f'COM{choice}'

if __name__ == '__main__':
    # Get the experiment name
    experiment_name_default = "test"
    experiment_name = int(input(f'Enter experiment name (press RETURN for "{experiment_name_default}"): ').strip() or experiment_name_default)

    # Get the duration of the valve opening
    duration_ms_default = 1000
    duration_ms = int(input(f'Enter valve opening duration (press RETURN for {duration_ms_default} ms): ').strip() or duration_ms_default)

    # Set the before and after times
    before_time = 2  # seconds
    after_time = 2  # seconds

    # Set up the serial connection
    arduino_port = find_serial_device(description='Adafruit')
    arduino_baudrate = 9600

    if arduino_port:
        ser = serial.Serial(arduino_port, arduino_baudrate, timeout=1)
        time.sleep(2)  # Wait for the connection to establish
    else:
        raise SystemError('Arduino not found')

    # Record the start time
    start_time = datetime.datetime.now(datetime.timezone.utc)
    readings = []

    # Start the while loop
    valve_opened = False
    finished_received = False
    loop_start_time = time.time()

    while True:
        current_time = time.time()
        elapsed_time = current_time - loop_start_time

        # Ask the Arduino for a single pressure readout
        ser.write('?PRESSURE\n'.encode())
        if ser.in_waiting > 0:
            pressure_value = ser.readline().decode('utf-8').rstrip()
            # TODO: Read out flow meter from serial
            flow_meter_value = 0
            readings.append((elapsed_time, pressure_value, flow_meter_value))

        # After 2 seconds, send a command to the Arduino to open the valve
        if not valve_opened and elapsed_time >= before_time:
            ser.write(f'OPEN {duration_ms}\n'.encode())
            valve_opened = True

        # See if the Arduino has sent a command "!FINISHED"
        if ser.in_waiting > 0:
            response = ser.readline().decode('utf-8').rstrip()
            if response == "!FINISHED":
                finished_received = True
                finished_time = current_time

        # Continue the loop for an additional time after receiving "!FINISHED"
        if finished_received and (current_time - finished_time) >= after_time:
            break

    # Save the readings to a CSV file
    timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M')
    filename = f'{timestamp}_{experiment_name}.csv'
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header
        csvwriter.writerow(['Experiment Name', experiment_name])
        csvwriter.writerow(['Start Time (UTC)', start_time])
        csvwriter.writerow(['Opening Duration (ms)', duration_ms])
        csvwriter.writerow(['Before Time (s)', before_time])
        csvwriter.writerow(['After Time (s)', after_time])
        csvwriter.writerow([])
        csvwriter.writerow(['Elapsed Time (s)', 'Pressure', 'Flow Meter'])

        # Write the readings
        for reading in readings:
            csvwriter.writerow(reading)


    # while True:
    #     if ser.in_waiting > 0:
    #         line = ser.readline().decode('utf-8').rstrip()
    #         print(f'\r{line}\nEnter command: ', end='')

    # def manual_input():
    #     while True:
    #         command = input("Enter command: ")
    #         ser.write(f'{command}\n'.encode())
    #
    # # Start the serial monitor in a separate thread
    # serial_thread = threading.Thread(target=monitor_serial)
    # serial_thread.daemon = True
    # serial_thread.start()
    #
    # # Start the manual input loop
    # manual_input()
    #
