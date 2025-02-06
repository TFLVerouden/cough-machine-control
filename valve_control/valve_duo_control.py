import propar
import serial
import time
import serial.tools.list_ports
import datetime
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

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
    # Ask if the user wants to save the data
    save_default = "y"
    save = (input('Do you want to save the data? (y/n): ').strip().lower()
            or save_default)
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
    
    # Do you want a secondary cough
    second_cough_default = "n"
    second_cough = (input('Do you want a second cough? (y/n): ').strip().lower()
            or second_cough_default)

    if second_cough == "y":
        #duration between coughs
        cough_pause_default = 500 #ms
        cough_pause = (input(f'How long of a wait between coughs? (press ENTER for 
                    {cough_pause_default} ms): )): ').strip()
            or second_cough_default)
        
        #Second_opening_duration
        second_opening_duration_default = 500 #ms
        second_opening_duration = (input(f'How long should the second opening take? (press ENTER for 
                    {second_opening_duration_default} ms): )): ').strip()
            or second_opening_duration_default)
        
        
        


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
            decoding_time = time.time_ns()
            response = ser.readline().decode('utf-8').rstrip()
            if response == "":
                no_response_time = time.time_ns()
                print(f"No response took {(no_response_time - decoding_time)} mseconds")

                continue
            elif response == "!":
                print('Valve closed')
                finished_received = True
                finished_time = current_time
                closing_valve_time = time.time_ns()
                print(f"Closing valve took {(closing_valve_time - decoding_time)} seconds")

            else:
                # Assume it's a pressure value
                pressure_value = float(response)
                # Read the flow meter value
                pre_flow_meter_time = time.time_ns()
                flow_meter_value = float(flow_meter.readParameter(8))
                flow_meter_time = time.time_ns()
                readings = np.append(readings, [current_time,
                                                pressure_value, flow_meter_value])
                pressure_value_time = time.time_ns()
                print(f"reading all sensors"
                      f" {(pressure_value_time - decoding_time)*1E-6} milliseconds")
                print(f" Only Flow took"
                      f" {(flow_meter_time - pre_flow_meter_time) * 1E-6} milliseconds")
        # Ask the Arduino for a single pressure readout

        ser.write('P?\n'.encode())

        # After a set time, send a command to the Arduino to open the valve
        if not valve_opened and elapsed_time >= (before_time_ms / 1000):
            opening_time = time.time()
            print('Opening valve...')
            ser.write(f'O {duration_ms}\n'.encode())
            valve_opened = True

        # Continue the loop for an additional time after receiving "!"
        if finished_received:
            if second_cough == "y":
                break
            elif (current_time - finished_time) >= (
                after_time_ms / 1000) and second_cough != "y":
                # Todo: If, for whatever reason, no "!" is received, the loop
                #  will continue indefinitely
                print('Experiment finished')
                break
    if second_cough == "y":
        
        print("Second cough")
        while True:
            # TODO: Speed up while loop. Problem seems to be waiting for
            #  an Arduino response
            current_time = time.time()
            waiting_time = current_time - finished_time
            elapsed_time = current_time - loop_start_time

            # Listen for commands from the Arduino
            if ser.in_waiting > 0:
                decoding_time = time.time_ns()
                response = ser.readline().decode('utf-8').rstrip()
                if response == "":
                    no_response_time = time.time_ns()
                    print(f"No response took {(no_response_time - decoding_time)} mseconds")

                    continue
                elif response == "!":
                    print('Valve closed')
                    finished_received = True
                    finished_time = current_time
                    closing_valve_time = time.time_ns()
                    print(f"Closing valve took {(closing_valve_time - decoding_time)} seconds")

                else:
                    # Assume it's a pressure value
                    pressure_value = float(response)
                    # Read the flow meter value
                    pre_flow_meter_time = time.time_ns()
                    flow_meter_value = float(flow_meter.readParameter(8))
                    flow_meter_time = time.time_ns()
                    readings = np.append(readings, [current_time,
                                                    pressure_value, flow_meter_value])
                    pressure_value_time = time.time_ns()
                    print(f"reading all sensors"
                            f" {(pressure_value_time - decoding_time)*1E-6} milliseconds")
                    print(f" Only Flow took"
                            f" {(flow_meter_time - pre_flow_meter_time) * 1E-6} milliseconds")
            # Ask the Arduino for a single pressure readout

            ser.write('P?\n'.encode())

            # After a set time, send a command to the Arduino to open the valve
            if not valve_opened and waiting_time >= (cough_pause / 1000):
                opening_time = time.time()
                print('Opening valve...')
                ser.write(f'O {second_opening_duration}\n'.encode())
                valve_opened = True

            # Continue the loop for an additional time after receiving "!"
            if finished_received:
                if (current_time - finished_time) >= (
                    after_time_ms / 1000):
                    # Todo: If, for whatever reason, no "!" is received, the loop
                    #  will continue indefinitely
                    print('Experiment finished')
                    break

if save == "y":
    # Save the readings to a CSV file
    print('Saving data...')
    flow_meter_calibration_value = float(10 / 30000) #L/s at maximum capacity: 30.000 a.u.
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
#### plotting
    plotname = os.path.join(data_dir, f'{timestamp}_{experiment_name}.png')
    plotdata= readings[8:,:]
    dt = np.diff(plotdata[:,0])
    mask = plotdata[:,2]>0 #finds the first time the flow rate is above 0
    t0 = plotdata[mask,0][0]
    peak_ind = np.argmax(plotdata[:,2])
    PVT = plotdata[peak_ind,0] - t0 #Peak velocity time
    CFPR = plotdata[peak_ind,2] #Critical flow pressure rate (L/s)
    CEV = np.sum(dt * plotdata[1:,2]) #Cumulative expired volume
    plotdata = plotdata[mask,:]
    t = plotdata[:,0] - opening_time
    fig, ax1 = plt.subplots()
    ax1.plot(t, plotdata[:,2], 'b-')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Flow rate (L/s)', color='b')
    ax1.set_title(f'Experiment: {experiment_name}, open: {duration_ms} ms, CFPR: {CFPR:.2f} L/s, PVT: {PVT:.2f} s, CEV: {CEV:.2f} L')
    ax1.grid()
    plt.savefig(plotname)