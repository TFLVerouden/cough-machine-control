import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import tkinter as tk
from tkinter import filedialog
from matplotlib.colors import LogNorm

#we use this fur future compatibility
pd.set_option('future.no_silent_downcasting', True)

#nice style
matplotlib.use("TkAgg")  # Or "Agg", "Qt5Agg", "QtAgg"
plt.rcParams.update({'font.size': 14})

#FINDING THE FILES
cwd = os.path.dirname(os.path.abspath(__file__))

path = os.path.join(cwd,"individual_data_files")
save_path = os.path.join(cwd,"results_spraytec")

txt_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')]

#NOW WE CHOSE with a dialog!!!!
root = tk.Tk()
root.withdraw()  # hide the root window

# Let user pick a file inside your directory
file = filedialog.askopenfilename(initialdir=path, filetypes=[("Text files", "*.txt")])

print("You picked:", file)
filename = file.split('/')[-1].replace('.txt', '')
print(filename)

#From here we read the data

df = pd.read_table(file,delimiter=",", encoding="latin-1")
df = df.replace('-', 0)
print(df.loc[0,"Date-Time"])
for col in df.columns:
    # Try converting each column to numeric, coercing errors to NaN
    df[col] = pd.to_numeric(df[col], errors='ignore')
important_columns= ["Date-Time","Transmission", "Duration","Time (relative)"]

columns_scattervalues = df.loc[:,"0.10000020":"1000.00195313"].columns.tolist()
#df[columns_scattervalues].astype(float)
important_columns = important_columns + columns_scattervalues
df_filtered= df.loc[:,important_columns]

####TEST

#time depended variables
time_chosen = 1

date= df_filtered.loc[time_chosen,"Date-Time"]
percentages = df_filtered.loc[time_chosen,columns_scattervalues]
t_start = df_filtered.loc[time_chosen,"Time (relative)"]
t_end = t_start + df_filtered.loc[time_chosen,"Duration"]
transmission = df_filtered.loc[time_chosen,"Transmission"]

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

### VOLUME PERCENTAGES

plt.figure(figsize=(9,6))
plt.bar(bin_edges[:-1], percentages, width=bin_widths, align='edge', edgecolor='black')

# Add labels
plt.xlabel(r"Diameter ($\mu$m)")
plt.ylabel("Volume Percentage (%)")
plt.title(f"Particle distribution at {date}, \n t= {round(t_start*1000)} to {round(t_end*1000)} ms, transmission: {transmission:.1f} % ")
plt.xscale('log')
plt.yscale('log')
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.ylim(0,40)
plt.xlim(bin_edges[0],bin_edges[-1])
# plt.show()


#NUMBER PERCENTAGES
n_percentages = percentages/ (bin_centers*1E-6)**3
n_percentages  = n_percentages/ sum(n_percentages)*100
plt.figure(figsize=(9,6))
plt.bar(bin_edges[:-1], n_percentages, width=bin_widths, align='edge', edgecolor='black')

# Add labels
plt.xlabel(r"Diameter ($\mu$m)")
plt.ylabel("Number Percentage (%)")
plt.title(f"Particle distribution at {date}, \n t= {round(t_start*1000)} to {round(t_end*1000)} ms, transmission: {transmission:.1f} % ")
plt.xscale('log')
plt.yscale('log')
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.ylim(0,40)
plt.xlim(bin_edges[0],bin_edges[-1])
# plt.show()

#Over time plot
len_arr= df_filtered.shape[0]
dates = np.array([],dtype=str)
percentages_all = np.zeros((len(columns_scattervalues),len_arr))

times = np.zeros(len_arr)
transmissions = np.zeros(len_arr)
print(dates)
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
for i in df_filtered.index:


    date= df_filtered.loc[i,"Date-Time"]
    dates = np.append(dates,date)
    percentages = df_filtered.loc[i,columns_scattervalues].values
    percentages_all[:,i] = percentages
    t_start = df_filtered.loc[i,"Time (relative)"]
    times[i] = t_start
    t_end = t_start + df_filtered.loc[i,"Duration"]
    transmission = df_filtered.loc[i,"Transmission"]
    transmissions[i] = transmission


### figure
# Set color limits (adjust as needed)
# vmin = 1e-1
# vmax = 5e1

# extent = [0, 0.25, bin_centers[0], bin_centers[-1]]

# fig,ax = plt.subplots()
# im =ax.imshow(percentages_all,cmap='grey_r',extent=extent,aspect='auto',origin='lower',norm=LogNorm(vmin=vmin, vmax=vmax))
# num_ticks = 10

# # X-axis: time
# x_ticks = np.linspace(extent[0], extent[1], num=num_ticks)
# ax.set_xticks(x_ticks)
# ax.set_xticklabels([f"{x:.2f}" for x in x_ticks])

# # Y-axis: diameters
# y_ticks = np.linspace(extent[2], extent[3], num=num_ticks)
# ax.set_yticks(y_ticks)
# ax.set_yticklabels([f"{y:.2f}" for y in y_ticks])
# # ax.set_yscale('log')
# # log_ticks = np.geomspace(bin_centers[0], bin_centers[-1], num=8)
# # ax.set_yticks(log_ticks)
# # ax.set_yticklabels([f"{t:.2f}" for t in log_ticks])

# ax.set_xlabel("Time (s)")
# ax.set_ylabel(r"Diameter ($\mu$m)")

# cbar = plt.colorbar(im, ax=ax)
# cbar.set_label("PDF (Volume)")

# #plt.colorbar()
# plt.grid(which= "both")
# plt.show()

def edges_from_centers(centers):
    edges = np.zeros(len(centers) + 1)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2
    edges[0] = centers[0] - (centers[1] - centers[0]) / 2
    edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2
    return edges

time_edges = np.append(times,t_end)
diameter_edges = edges_from_centers(bin_centers)

# Make meshgrid of edges
X, Y = np.meshgrid(time_edges, diameter_edges)

vmin = 0
vmax = 40

fig, ax = plt.subplots(figsize=(9,6))

pcm = ax.pcolormesh(X, Y, percentages_all,
                    norm=LogNorm(vmin=1e-1, vmax=5e1),
                    cmap= 'grey_r')

ax.set_yscale('log')
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'Diameter ($\mu$m)')
ax.set_xlim(0,0.2)
ax.set_ylim(bin_centers[0], bin_centers[-1])


ax2 = ax.twinx()
ax2.plot(times,100-transmissions,c="r")
ax2.set_ylim(0,100)
ax2.set_ylabel("Reflected (%)")
cbar = plt.colorbar(pcm, ax=ax)

cbar.set_label('PDF (Volume Percentage)')
pos = cbar.ax.get_position()  # get current position [x0, y0, width, height]

# Move the colorbar right by increasing x0 and x1:
new_pos = [pos.x0 + 0.05, pos.y0, pos.width, pos.height]
cbar.ax.set_position(new_pos)
full_save_path = os.path.join(save_path,filename)
plt.savefig(full_save_path+".png")
# plt.show()

