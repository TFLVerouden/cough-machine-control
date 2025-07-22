import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import tkinter as tk
from tkinter import filedialog
#we use this fur future compatibility
pd.set_option('future.no_silent_downcasting', True)

#nice style
matplotlib.use("TkAgg")  # Or "Agg", "Qt5Agg", "QtAgg"

#FINDING THE FILES
cwd = os.path.dirname(os.path.abspath(__file__))

path = os.path.join(cwd,"individual_data_files")

txt_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')]

#NOW WE CHOSE with a dialog!!!!
root = tk.Tk()
root.withdraw()  # hide the root window

# Let user pick a file inside your directory
file = filedialog.askopenfilename(initialdir=path, filetypes=[("Text files", "*.txt")])

print("You picked:", file)

#From here we read the data
df = pd.read_table(file,delimiter=",")
df = df.replace('-', 0)
for col in df.columns:
    # Try converting each column to numeric, coercing errors to NaN
    df[col] = pd.to_numeric(df[col], errors='coerce')
important_columns= ["Date-Time","Transmission", "Duration","Time (relative)"]

columns_scattervalues = df.loc[:,"0.10000020":"1000.00195313"].columns.tolist()
#df[columns_scattervalues].astype(float)
important_columns = important_columns + columns_scattervalues
df_filtered= df.loc[:,important_columns]

####TEST

#time depended variables
percentages = df_filtered.loc[0,columns_scattervalues]
t_start = df_filtered.loc[0,"Time (relative)"]
t_end = t_start + df_filtered.loc[0,"Duration"]
transmission = df_filtered.loc[0,"Transmission"]

###Extracting
bin_centers = np.array(columns_scattervalues,dtype=float)
diffs = np.diff(bin_centers)
bin_edges = np.zeros(len(bin_centers) + 1)

bin_edges[1:-1] = (bin_centers[:-1] + bin_centers[1:]) / 2
# First edge (extrapolate)
bin_edges[0] = bin_centers[0] - diffs[0] / 2

# Last edge (extrapolate)
bin_edges[-1] = bin_centers[-1] + diffs[-1] / 2
# Step 2: Calculate widths
bin_widths = np.diff(bin_edges)

#plotting


plt.figure()
plt.bar(bin_edges[:-1], percentages, width=bin_widths, align='edge', edgecolor='black')

# Add labels
plt.xlabel(r"Diameter ($\mu$m)")
plt.ylabel("Number Percentage (%)")
plt.title(f"Particle distribution, t= {round(t_start*1000)} to {round(t_end*1000)} ms, transmission: {transmission:.1f} % ")
plt.xscale('log')
plt.grid()
plt.ylim(0,40)
plt.xlim(bin_edges[0],bin_edges[-1])
plt.show()