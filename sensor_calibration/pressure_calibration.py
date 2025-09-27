import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load the calibration data
data_file = 'pressure_calibration_data.txt'
data = np.loadtxt(data_file, delimiter=',', skiprows=1)  # Comma-separated, skip header
pressure_values = data[:, 0]  # First column: pressure values
sensor_readings = data[:, 1]  # Second column: sensor readings

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(sensor_readings, pressure_values)

# Calculate the conversion factor (slope)
conversion_factor = slope
print(f"Conversion: p = {conversion_factor:.4f} bar/mA * I - {-intercept:.4f} bar")

# Generate the calibration plot
plt.figure(figsize=(8, 6))
plt.scatter(sensor_readings, pressure_values, label='Data', color='blue', edgecolor='k')
plt.plot(sensor_readings, slope * sensor_readings + intercept, label='Fit: p = {:.4f}I + {:.4f}'.format(slope, intercept), color='red')
plt.xlabel('Sensor reading (mA)')
plt.ylabel('Pressure (bar)')
plt.title('Pressure sensor calibration')
plt.legend()
plt.grid()
plt.tight_layout()

# Save and show the plot
plt.savefig('pressure_calibration_plot.png')
# plt.show()