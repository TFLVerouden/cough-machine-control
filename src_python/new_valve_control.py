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

# from functions import Gupta2009 as Gupta

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
    ser.reset_input_buffer()
    ser.write('T?\n'.encode())
    time.sleep(0.1) #wait for the response
    Temperature = ser.readline().decode('utf-8', errors='ignore').rstrip()
    RH= ser.readline().decode('utf-8', errors='ignore').rstrip()
    Temperature = Temperature.lstrip('T')
    RH = RH.lstrip('RH')

    if verbose:
        print(f'Temperature: {Temperature} °C; relative humidity: {RH} %')
    return RH, Temperature

def reading_pressure(verbose=False):
    ser.reset_input_buffer()
    ser.write('P?\n'.encode())
    time.sleep(0.1) #wait for the response
    pressure = ser.readline().decode('utf-8', errors='ignore').rstrip()
    pressure_value = pressure.lstrip('P')

    if verbose:
        print(f'Pressure: {pressure_value} mbar')
    return pressure_value

def extract_csv_dataset(filename, delimiter=','):
    # Define output variables
    time = []
    mA = []
    row_idx = 0

    with open(filename, 'r') as csvfile:
        # Reader object of the file
        csvreader = csv.reader(csvfile, delimiter=delimiter)

        # extract data from the file and create time and value arrays
        for rows in csvreader:
            if not rows[0] or not rows[1]:
                print(f"Encountered empty cell in flow profile dataset, row index {row_idx}!")
                time = []
                mA = []
                break
            else:
                # replace ',' with '.' depending on csv format (';' delim vs ',' delim)
                time.append(rows[0].replace(',', '.'))
                mA.append(rows[1].replace(',', '.'))
                row_idx += 1

    return time, mA

def format_csv_dataset(time_array, mA_array, prefix="LOAD", handshake_delim=" ", data_delim=",", line_feed='\n'):

    duration = time_array[-1]

    # Check for inconsistent dataset length
    if len(time_array) != len(mA_array) or len(time_array) == 0 or len(mA_array) == 0:
        print(
            f"Arrays are not compatible! Time length: {len(time_array)}, mA length: {len(mA_array)}")
        return
    else:
        # Create 'handshake' sequence
        header = [prefix, handshake_delim, str(
            len(time_array)), handshake_delim, duration, handshake_delim]

        # Append timestamps and values in order <time0, mA0, time1, mA1, time2, mA2>
        data = [str(val) for time, mA in zip(time_array, mA_array)
                for val in (time, mA)]

        # format into one string to send over serial
        output = "".join(header) + data_delim.join(data) + line_feed

    return output

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
                    height = line.split(b': ')[1].strip().decode('utf-8', errors='ignore')
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

def manual_mode():
    print("\n=== MANUAL MODE ===")
    print("Enter commands to send to MCU (type 'exit' to return to main menu)\n")

    while True:
        cmd = input("Enter command: ").strip()

        if cmd.lower() == 'exit':
            next_step_default = "q"
            next_step = (input("What to do next - continue to experimental mode or quit? (e/q): ").strip().lower() or next_step_default)
            if next_step == 'q':
                print("Exiting program.")
                exit()
            elif next_step == 'e':
                break
        else:
            ser.write((cmd + '\n').encode('utf-8'))
            time.sleep(0.1)  # wait for response
            while ser.in_waiting > 0:
                response = ser.readline().decode('utf-8', errors='ignore').rstrip()
                print(f"Response: {response}")

def send_dataset():
    default_delimiter = ','
    delimiter = (input(f'Enter CSV delimiter (press ENTER for "{default_delimiter}"): ').strip() or default_delimiter)

    # Defining defaul file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_filepath = os.path.join(script_dir,'drawn_curve.csv')

    while True:
        filename = (input(f'Enter dataset filename (press ENTER for "{default_filepath}"): ').strip() or default_filepath)
        try:
            data = extract_csv_dataset(filename, delimiter)
            print(f'Sending dataset from file: {filename}')
            break  # Exit loop if file is successfully read
        except FileNotFoundError:
            print(f'Error: File "{filename}" not found. Please try again.')
    
    serial_command = format_csv_dataset(data[0], data[1])
    ser.write(serial_command.encode('utf-8'))

def verify_mcu_dataset_received_with_timeout(expected_msg="DATASET_RECEIVED", timeout_sec=5):
    start_time = time.time()
    
    while (time.time() - start_time) < timeout_sec:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line == expected_msg:
                print("MCU confirmed dataset receipt.")
                return True
        time.sleep(0.1)  # Small sleep to reduce CPU usage
        
    print("Error: MCU confirmation timed out.")
    return False

def retreive_experiment_data(filename, experiment_name, start_time, end_time, Temperature, RH, height):
    
    started = False

    directory = "C:\\CoughMachineData"
    os.makedirs(directory, exist_ok=True)
    full_path = os.path.join(directory, filename)

    with open(full_path, "w") as f:
        f.write(f"Experiment Name,{experiment_name}\n")
        f.write(f"Start Time (UTC),{start_time.isoformat()}\n")
        f.write(f"End Time (UTC),{end_time.isoformat()}\n")
        f.write(f"Temperature (C),{Temperature}\n")
        f.write(f"Relative Humidity (%),{RH}\n")
        f.write(f"Lift Height (mm),{height}\n")

        ser.write('F\n'.encode())  

        while True:
            raw_line = ser.readline()
            try:
                line = raw_line.decode('utf-8')
            except UnicodeDecodeError:
                continue

            clean_line = line.strip()

            if "START_OF_FILE" in clean_line:
                started = True
                continue
            elif "END_OF_FILE" in clean_line:
                break
            if started:
                f.write(line)
    print(f"Experiment data saved to {filename} in {os.path.abspath(filename)}")

if __name__ == '__main__':

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

    # Create the data directory if it doesn't exist
    Spraytec_data_saved_check()
    # Create the data directory if it doesn't exist
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Ask which mode to use
    mode_default = "m"
    mode = (input("Select mode - manual or experimental? (m/e): ").strip().lower() or mode_default)

    if mode == 'm':
        manual_mode()
    elif mode == "e":
        print("\n=== EXPERIMENT MODE ===")

    # Ask if the user wants to save the data
    save_default = "n"
    save = (input(f'Do you want to save the experimental data (press ENTER for {save_default})? (y/n): ').strip().lower()
            or save_default)
    
    # Get the experiment name
    if save == "y":
        experiment_name_default = "test"
        experiment_name = (input(f'Enter experiment name (press ENTER for '
                                f'"{experiment_name_default}"): ').strip()
                        or experiment_name_default)

    #Processing compare to model
    if save == "y":
        model_default = "n"
        model = (input('Do you want to include the Gupta model in the data (press ENTER for 'f'{model_default})? (y/n): ').strip().lower()
                or model_default)
    
    # Connect to SprayTec lift if available
    lift_port = find_serial_device(description='Mega', continue_on_error=True)
    if lift_port:
        lift = SprayTecLift(lift_port)
    else:
        print('Warning: SprayTec lift not found; height will not be recorded.')

    # Ask if the user wants to load a dataset
    load_dataset_default = "n"
    load_dataset = (input(f'Do you want to upload a flow curve (press ENTER for {load_dataset_default})? (y/n): ').strip().lower() or load_dataset_default)
    if load_dataset == 'y':
        send_dataset()

        # Immediately check for the confirmation
        if verify_mcu_dataset_received_with_timeout():
            print("Proceeding to valve control phase.")
        else:
            print("Failed to sync with MCU. Aborting.")
            sys.exit(1)
    elif load_dataset == 'n':
        print("Proceeding without loading a dataset.")

    default_pressure = 1
    pressure = (input(f'Enter target tank pressure in bar (press ENTER for {default_pressure} bar): ').strip() or str(default_pressure))
    ser.write(f'SP {pressure}\n'.encode())

    # Ask if ready to execute experiment
    while True:
        ready_default = "n"
        ready = (input('Ready to start the experiment (press ENTER for {ready_default})? (y/n): ').strip().lower() or ready_default)
        if ready == 'y':
            ser.write("RUN\n".encode())
            time.sleep(0.1)  # wait for response
            while ser.in_waiting > 0:
                response = ser.readline().decode('utf-8', errors='ignore').rstrip()
                if response == "EXECUTING_DATASET":
                    print("MCU has started executing the dataset.")
                else:
                    print(f"Something went wrong, response: {response}")
            break
        else:
            print('Take your time. Press y when ready.')


    # Take humidity, temprature, pressure readings and lift height readings
    RH, Temperature = reading_temperature()

    if lift_port:
        height = lift.get_height()
    else:
        height = np.nan

    start_time = datetime.datetime.now(datetime.timezone.utc)
    loop_start_time = time.time()
    print('Starting experiment...')

    starting_experiment = True
    finished_experiment = False

    while True:

        if ser.in_waiting > 0:
            response = ser.readline().decode('utf-8', errors='ignore').rstrip()

            if response == "DONE_SAVING_TO_FLASH":
                end_time = datetime.datetime.now(datetime.timezone.utc)
                starting_experiment = False
                finished_experiment = True
                if save == 'y':
                    print("Saved experiment detected. Starting file retrieval...")

                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"{experiment_name}_{timestamp}.csv"

                    retreive_experiment_data(filename, experiment_name, start_time, end_time, Temperature, RH, height)
                else:
                    print("Experiment completed. Data not saved as per user choice.")

        # Break the loop if experiment is finished
        if finished_experiment:
            ser.close()
            if lift_port:
                lift.close_connection()
            
            print("Serial connections closed")
            break