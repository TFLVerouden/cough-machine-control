import csv
import serial

# Configure variables
serial_port = "COM14"
baud = 115200
filename = "drawn_curve.csv"
delimiter = ';'

# =========================================================================================
# Extract time and value arrays from csv file. Input: "filename", "delimiter"
# =========================================================================================


def extract(filename, delimiter=','):
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
                print(f"Encountered empty cell at row index {row_idx}!")
                time = []
                mA = []
                break
            else:
                # replace ',' with '.' depending on csv format (';' delim vs ',' delim)
                time.append(rows[0].replace(',', '.'))
                mA.append(rows[1].replace(',', '.'))
                row_idx += 1

    return time, mA

# =========================================================================================
# Format incomming time and value arrays according to serial protocol. Input: [time], [mA]
# =========================================================================================


def format(time_array, mA_array, prefix="LOAD", handshake_delim=" ", data_delim=",", line_feed='\n'):

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


# =========================================================================================
# Write dataset over serial using communication protocol
# =========================================================================================
sc = serial.Serial(serial_port, baud, timeout=1)

# data[0] is time array, data[1] is mA array
data = extract(filename, delimiter)
serial_command = format(data[0], data[1])

print(serial_command)                       # Debug print

# Encode serial command to utf8 format for arduino.
sc.write(serial_command.encode('utf8'))
