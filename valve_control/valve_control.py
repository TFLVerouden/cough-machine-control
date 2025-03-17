import propar
import serial
import time
import serial.tools.list_ports
import datetime
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

cwd = os.path.abspath(os.path.dirname(__file__))

parent_dir = os.path.dirname(cwd)
print(parent_dir)
#function_dir = os.path.join(parent_dir, 'cough-machine-control')
function_dir = os.path.join(parent_dir,'functions')
print(function_dir)
sys.path.append(function_dir)
import Gupta2009 as Gupta
import pumpy 
from Ximea import Ximea


####Finished loading Modules



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

def reading_temperature():
    ser.write('T?\n'.encode())
    time.sleep(0.1) #wait for the response
    Temperature = ser.readline().decode('utf-8').rstrip()
    RH= ser.readline().decode('utf-8').rstrip()
    Temperature = Temperature.lstrip('T')
    RH = RH.lstrip('RH')
    return RH, Temperature

if __name__ == '__main__':
    # Create the data directory if it doesn't exist
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Show Ximea cam live
    try:
        cam = Ximea(export_folder = data_dir)
        cam.set_param(exposure=100)
        cam.live_view(before=True)
    except Exception as e:
        print("Ximea not found")
    
    # Ask if the user wants to save the data

    save_default = "y"
    save = (input('Do you want to save the data? (y/n): ').strip().lower()
            or save_default)
    # Get the experiment name
    experiment_name_default = "test"
    if save == "y":
        experiment_name = (input(f'Enter experiment name (press ENTER for '
                                f'"{experiment_name_default}"): ').strip()
                        or experiment_name_default)

    # Get the duration of the valve opening
    duration_ms_default = 50
    duration_ms = int(input(f'Enter valve opening duration (press ENTER for '
                            f'{duration_ms_default} ms): ').strip()
                      or duration_ms_default)
    #Processing compare to model
    if save == "y":
        model_default = "n"
        model = (input('Do you want to include the model in the data? (y/n): ').strip().lower()
                or model_default)

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
    flow_meter_baudrate = 115200
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

    readings = np.array([],dtype=float)
    # Start the while loop
    valve_opened = False
    finished_received = False
    #To make it easier to time with the camera
    time.sleep(0.5)
    ready = (input('Ready to cough?'))
    # Record the start time
    time.sleep(0.5)
    start_time = datetime.datetime.now(datetime.timezone.utc)

    print('Starting experiment...')
    #We are going to send a command to the Arduino to measure the temperature
    #and relative humidity of the environment.
    #These lines send the command to read Temperature to arduino
    #It receives two responses. To make sure that we are not interfering anything
    #we are going to lstrip with their signature characters

    RH, Temperature = reading_temperature()
    loop_start_time = time.time()
    while True:
        current_time = time.time()
        elapsed_time = current_time - loop_start_time

        # Listen for commands from the Arduino
        if ser.in_waiting > 0:
            response = ser.readline().decode('utf-8').rstrip()
            if response == "":
                continue
            elif response == "!":
                print('Valve closed')
                finished_received = True
                finished_time = current_time
            else:
                # Assume it's a pressure value
                pressure_value = float(response)
                # Read the flow meter value
                flow_meter_value = float(flow_meter.readParameter(8))
                readings = np.append(readings, [current_time,
                                                pressure_value, flow_meter_value])

        # Ask the Arduino for a single pressure readout

        ser.write('P?\n'.encode())

        # After a set time, send a command to the Arduino to open the valve
        if not valve_opened and elapsed_time >= (before_time_ms / 1000):
            print('Opening valve...')
            ser.write(f'O {duration_ms}\n'.encode())
            valve_opening_time = time.time()
            valve_opened = True

        # Continue the loop for an additional time after receiving "!"
        if finished_received and (current_time - finished_time) >= (
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
    readings[:,0] = readings[:,0] -valve_opening_time #time since the valve opened
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
        csvwriter.writerow(['Ambient Temperature (Celsius)', Temperature])
        csvwriter.writerow(['Relative Humidity (%)', RH])
        csvwriter.writerow([])
        csvwriter.writerow(
                ['Elapsed time (s)', 'Pressure (bar)', 'Flow rate (L/s)'])

        # Write the readings
        for reading in readings:
            csvwriter.writerow(reading)
#### plotting
    plotname = os.path.join(data_dir, f'{timestamp}_{experiment_name}.png')
    plotdata= readings
    dt = np.diff(plotdata[:,0])
    mask = plotdata[:,2]>0 #finds the first time the flow rate is above 0
    mask_opening = plotdata[:,0]>0 #finds the first time the valve is opened
    t0 = plotdata[mask,0][0]
    peak_ind = np.argmax(plotdata[:,2])
    PVT = plotdata[peak_ind,0] - t0 #Peak velocity time
    CFPR = plotdata[peak_ind,2] #Critical flow pressure rate (L/s)
    CEV = np.sum(dt * plotdata[1:,2]) #Cumulative expired volume
    plotdata = plotdata[mask_opening,:]
    t = plotdata[:,0] -t0
    fig, ax1 = plt.subplots()
    ax1.plot(t, plotdata[:,2], 'b-',label= "Measurement",marker= "o",markeredgecolor= "k")
    if model == "y":
        #person E, me based on Gupta et al
        Tau = np.linspace(0,10,101)

        PVT_E, CPFR_E, CEV_E = Gupta.estimator("Male",70, 1.89)

        cough_E = Gupta.M_model(Tau,PVT_E,CPFR_E,CEV_E)
        ax1.plot(Tau*PVT_E,cough_E* CPFR_E, 'r:',label= "Model")
    ax1.set_xlabel('Time (s)')
    ax1.legend()
    ax1.set_ylabel('Flow rate (L/s)')
    ax1.set_title(f'Exp: {experiment_name}, open: {duration_ms} ms \n'
                  f' CFPR: {CFPR:.1f} L/s, PVT: {PVT:.2f} s, CEV: {CEV:.1f} L\n'
                  f'T: {Temperature} C, RH: {RH} %')
    ax1.grid()
    plt.tight_layout()
    plt.savefig(plotname)

try:
    cam.live_view(before=False)
except Exception as e:
        print("Ximea not found")