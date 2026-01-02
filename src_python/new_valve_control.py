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
import pumpy 
from functions import Gupta2009 as Gupta

cwd = os.path.abspath(os.path.dirname(__file__))

parent_dir = os.path.dirname(cwd)
print(parent_dir)
#function_dir = os.path.join(parent_dir, 'cough-machine-control')
function_dir = os.path.join(parent_dir,'functions')
print(function_dir)
sys.path.append(function_dir)

####Finished loading Modules
def split_array_by_header_marker(arr, marker='Date-Time'):
    arr = np.array(arr)
    header = arr[:,0]
    rows = arr[:,1:]

    # Find indices where header has the marker
    split_indices = [i for i, val in enumerate(header) if val == marker]
    split_indices.append(len(header))  # include end boundary

    result = []
    for i in range(len(split_indices) - 1):
        start = split_indices[i]
        end = split_indices[i+1]
        section = arr[start:end]
        result.append(section)

    return result

def Spraytec_data_saved_check():
    """
    This function saves the last spraytec measurement of the previous run to a .txt
    in the folder individual_data_files. Do not touch this if you do not know waht you are doing!
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_path = os.path.dirname(current_dir)  # one level up
    spraytec_path = os.path.join(parent_path,"spraytec")
    path = os.path.join(spraytec_path,"SPRAYTEC_APPEND_FILE.txt")
    save_path = os.path.join(spraytec_path, "individual_data_files")
    file = np.loadtxt(path,dtype=str,delimiter=',')
    split_sections = split_array_by_header_marker(file)
    last_file = split_sections[-1]
    time_created= last_file[1,0]
    filename= last_file[1,1]
    dt = datetime.datetime.strptime(time_created, '%d %b %Y %H:%M:%S.%f')
    # Format as YYYY_MM_DD_HH_MM
    file_name_time = dt.strftime('%Y_%m_%d_%H_%M')
    save_path = os.path.join(save_path,file_name_time +"_" +filename + ".txt")
    if not os.path.exists(save_path):
        np.savetxt(save_path,last_file,fmt='%s',delimiter=',')
        print(f"Saved spraytec_data of {file_name_time}")

def find_serial_device(description, continue_on_error=False):
    ports = list(serial.tools.list_ports.comports())
    ports.sort(key=lambda port: int(port.device.replace('COM', '')))

    # Filter ports where the description contains the provided keyword
    matching_ports = [port.device for port in ports if description in port.description]

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
    
def reading_temperature(verbose=False):
    ser.write('T?\n'.encode())
    time.sleep(0.1) #wait for the response
    Temperature = ser.readline().decode('utf-8').rstrip()
    RH= ser.readline().decode('utf-8').rstrip()
    Temperature = Temperature.lstrip('T')
    RH = RH.lstrip('RH')

    if verbose:
        print(f'Temperature: {Temperature} °C; relative humidity: {RH} %')
    return RH, Temperature

def reading_pressure(verbose=False):
    ser.write('P?\n'.encode())
    time.sleep(0.1) #wait for the response
    pressure = ser.readline().decode('utf-8').rstrip()
    pressure_value = pressure.lstrip('P')

    if verbose:
        print(f'Pressure: {pressure_value} mbar')
    return pressure_value

class SprayTecLift(serial.Serial):
    def __init__(self, port, baudrate=9600, timeout=1):
        super().__init__(port=port, baudrate=baudrate, timeout=timeout)
        time.sleep(1)  # Allow time for the connection to establish
        print(f"Connected to SprayTec lift on {port}")

    def get_height(self):
        """Send a command to get the platform height and parse the response."""
        try:
            self.write(b'?\n')  # Send the status command
            response = self.readlines()
            for line in response:
                if line.startswith(b'  Platform height [mm]: '):
                    height = line.split(b': ')[1].strip().decode('utf-8')
                    return float(height)
            print('Warning: No valid response containing "Platform height [mm]" was found.')
            return None
        except Exception as e:
            print(f"Error while reading lift height: {e}")
            return None

    def close_connection(self):
        """Close the serial connection."""
        self.close()
        print("Lift connection closed.")

if __name__ == '__main__':

    # Create the data directory if it doesn't exist
    Spraytec_data_saved_check()
    # Create the data directory if it doesn't exist
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Ask if the user wants to save the data
    save_default = "n"
    save = (input('Do you want to save the data? (y/n): ').strip().lower()
            or save_default)
    
    # Get the experiment name
    experiment_name_default = "test"
    if save == "y":
        experiment_name = (input(f'Enter experiment name (press ENTER for '
                                f'"{experiment_name_default}"): ').strip()
                        or experiment_name_default)

    #Processing compare to model
    if save == "y":
        model_default = "n"
        model = (input('Do you want to include the model in the data? (y/n): ').strip().lower()
                or model_default)
        
    # Set the before and after times
    before_time_ms = 0
    after_time_ms = 1000

    # Set up the Arduino serial connection
    arduino_port = find_serial_device(description='ItsyBitsy')
    arduino_baudrate = 115200
    if arduino_port:
        ser = serial.Serial(arduino_port, arduino_baudrate,
                            timeout=0)  # Non-blocking mode
        time.sleep(1)  # Wait for the connection to establish
        print(f'Connected to Arduino on {arduino_port}')
    else:
        raise SystemError('Arduino not found')
    
    # Readout SprayTec lift height; if not found, give warning message but continue
    lift_port = find_serial_device(description='Mega', continue_on_error=True)
    if lift_port:
        lift = SprayTecLift(lift_port)
    else:
        print('Warning: SprayTec lift not found; height will not be recorded.')

    # check if ready to start experiment
    ready = (input('Ready to cough?'))

    # Take humidity, temprature and pressure readings
    RH, Temperature = reading_temperature(verbose=True)
    pressure = reading_pressure(verbose=True)

    # Read out the lift height
    if lift_port:
        height = lift.get_height()
    else:
        height = np.nan

