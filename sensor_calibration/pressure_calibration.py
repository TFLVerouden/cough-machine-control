import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path
from tkinter import Tk, filedialog

from time_utils import timestamp_str

# Create a root window (hidden)
root = Tk()
root.withdraw()

# Get the directory where this script is located
script_dir = Path(__file__).resolve().parent
docs_calibration_dir = script_dir.parent / "docs" / "calibration"
docs_calibration_dir.mkdir(parents=True, exist_ok=True)

# Prompt user to select calibration file
data_file = filedialog.askopenfilename(
    title="Select calibration data file",
    initialdir=str(script_dir),
    filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
)

if not data_file:
    print("No file selected. Exiting.")
    exit()

# Create output subfolder in docs/calibration if it doesn't exist
output_folder = docs_calibration_dir
output_folder.mkdir(parents=True, exist_ok=True)

# Load the calibration data
# Comma-separated, skip header
data = np.loadtxt(data_file, delimiter=',', skiprows=1)
pressure_values = data[:, 0]  # First column: pressure values
sensor_readings = data[:, 1]  # Second column: sensor readings

# Extract base filename without extension for output filenames
base_filename = Path(data_file).stem
timestamp = timestamp_str()

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(
    sensor_readings, pressure_values)

# Calculate the conversion factor (slope)
conversion_factor = slope
print(
    f"Conversion: p = {conversion_factor:.4f} bar/mA * I - {-intercept:.4f} bar")

# Generate the calibration plot
plt.figure(figsize=(8, 6))
plt.scatter(sensor_readings, pressure_values,
            label='Data', color='blue', edgecolor='k')
plt.plot(sensor_readings, slope * sensor_readings + intercept,
         label='Fit: p = {:.4f}I + {:.4f}'.format(slope, intercept), color='red')
plt.xlabel('Sensor reading (mA)')
plt.ylabel('Pressure (bar)')
plt.title(f'Pressure sensor calibration - {base_filename}')
plt.legend()
plt.grid()
plt.tight_layout()

# Save the plot as PDF to output folder with timestamped name
output_plot = output_folder / f'{base_filename}_{timestamp}_plot.pdf'
plt.savefig(output_plot)

output_npz = output_folder / f'{base_filename}_{timestamp}_calibration.npz'
np.savez(output_npz,
         slope=slope,
         intercept=intercept,
         r_value=r_value,
         p_value=p_value,
         std_err=std_err,
         sensor_readings=sensor_readings,
         pressure_values=pressure_values)
print(f"Calibration saved to {output_npz}")
# plt.show()
