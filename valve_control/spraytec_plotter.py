import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")  # Or "Agg", "Qt5Agg", "QtAgg"
path = r"C:\Users\local2\PycharmProjects\cough-machine-control\experiments\Spraytec_data_31032025\1-1_PEO2M_1_numd_av.txt"

data = pd.read_csv(path, delimiter=",", skiprows=66, nrows=60,encoding="latin1",names=["PDF", "Cum", "Diameter"])
print(data["Diameter"])
bin_edges= (data["Diameter"][1:].values + data["Diameter"][:-1].values)/2
bin_edges = np.append(bin_edges, (data["Diameter"].values[-1]-bin_edges[-1])*2)  # Append the last bin edge
print(bin_edges)


bin_width = np.diff(bin_edges)
bin_width = np.append(bin_width, (data["Diameter"].values[-1]-bin_edges[-1])*2)  # Append the last bin width

print(np.shape(data["PDF"][1:]))



area = np.sum(data["Diameter"] * data["PDF"])  # Total area under the curve (sum of bin areas)
data["PDF"] /= area  # Normalize the PDF to make it a probability density function
plt.figure(figsize=(10, 6))
plt.bar(bin_edges, data["PDF"], label="PDF", color="blue",edgecolor="black", alpha=0.7, width=bin_width,align="edge")
plt.grid()
plt.xscale('log')
plt.xlabel("Diameter (µm)")
plt.ylabel("Probability Density (1/µm)")
plt.show()

