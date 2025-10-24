import os
"""
Produces the average plots of the spraytec data either via a loop over a keyphrase or via a file explorer
"""
keyphrase = "PEO_0dot03_1dot5ml_1dot5bar_80ms"  ##change this for different statistics

#FINDING THE FILES
cwd = os.path.dirname(os.path.abspath(__file__))

path = os.path.join(cwd,"Averages")
path = os.path.join(path,"Unweighted","0dot03") #for the unweighted ones
#path = os.path.join(path,"Weighted") #for the weighted ones
print(f"Path: {path}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

import tkinter as tk
from tkinter import filedialog
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import re
import sys
#we use this fur future compatibility
pd.set_option('future.no_silent_downcasting', True)

#nice style
matplotlib.use("TkAgg")  # Or "Agg", "Qt5Agg", "QtAgg"
plt.rcParams.update({'font.size': 14})
cwd = os.path.dirname(os.path.abspath(__file__))
print(cwd)
parent_dir = os.path.dirname(cwd)
function_dir = os.path.join(parent_dir,'functions')

sys.path.append(function_dir)
from cvd_check import set_cvd_friendly_colors

colors = set_cvd_friendly_colors()
colors[3]="green"
colors[4]= "k"

save_path = os.path.join(cwd,"results_spraytec","Serie_Averages")
series_savepath = os.path.join(save_path,"npz_files")
os.makedirs(series_savepath, exist_ok=True)
print(f"Save path {save_path}")


txt_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')]
pattern = re.compile(rf"average_{re.escape(keyphrase)}_\d+(?:_.*)?\.txt")

# Filter matching files
matching_files = [f for f in txt_files if pattern.search(os.path.basename(f))and "skip" not in os.path.basename(f)]
save_path = os.path.join(save_path,keyphrase)

if not matching_files:
    print("none found")
    quit()
# Create folder if it doesn't exist
os.makedirs(save_path, exist_ok=True)


# #NOW WE CHOSE with a dialog!!!!
# root = tk.Tk()
# root.withdraw()  # hide the root window

# # Let user pick a file inside your directory
# file = filedialog.askopenfilename(initialdir=path, filetypes=[("Text files", "*.txt")])

# print("You picked:", file)
#### The loop over all files
weights =[]
total_n_percentages =[]
total_v_percentages= []
for file in matching_files:

    filename = file.split('\\')[-1].replace('.txt', '')
    filename =filename.split('ms')[0]
    #From here we read the data

    df = pd.read_table(file,delimiter=",",encoding='latin1')
    df = df.replace('-', 0)

    
    for col in df.columns:
        # Try converting each column to numeric, coercing errors to NaN
        df[col] = pd.to_numeric(df[col], errors='ignore')
    important_columns= ["Date-Time","Transmission", "Duration","Time (relative)","Number of records in average "]

    columns_scattervalues = df.loc[:,"% V (0.100-0.117µm)":"% V (857.698-1000.002µm)"].columns.tolist()


    important_columns = important_columns + columns_scattervalues
    
    df_filtered= df.loc[:,important_columns]



    #time depended variables


    date= df_filtered.loc[0,"Date-Time"]

    percentages = df_filtered.loc[0,columns_scattervalues]
    t_end = df_filtered.loc[0,"Time (relative)"]
    t_start = t_end - df_filtered.loc[0,"Duration"]
    transmission = df_filtered.loc[0,"Transmission"]
    num_records = df_filtered.loc[0,"Number of records in average "]
    weights.append(num_records)
    total_v_percentages.append(percentages.values)
    ###Extracting
    bin_centers = np.array([])
    for column in columns_scattervalues:
        match = re.search(r"\(([\d.]+)-([\d.]+)", column)
        if match:
            lower = float(match.group(1))
            upper = float(match.group(2))
            center = (lower + upper) / 2
            bin_centers = np.append(bin_centers,center)





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




    #NUMBER PERCENTAGES
    n_percentages = percentages/ (bin_centers*1E-6)**3
    n_percentages  = n_percentages/ sum(n_percentages)*100
    total_n_percentages.append(n_percentages)


total_n_percentages =np.array(total_n_percentages)
total_v_percentages =np.array(total_v_percentages)


min_records =5
max_records = 200
weights= np.array(weights)
print(np.shape(total_n_percentages))
print(np.max(total_n_percentages,axis=1))

max_n_check = np.max(total_n_percentages,axis=1)
max_n_limit = 75
print(max_n_check<max_n_limit)
#weights = np.ones(len(weights))*10
mask = (weights> min_records) & (weights<max_records) & (max_n_check<max_n_limit)
print(mask)

#weights = np.ones(len(weights))*10
weights= weights[:, np.newaxis]
total_n_percentages=total_n_percentages[mask]
total_v_percentages = total_v_percentages[mask]

total_n_percentages =np.sum(total_n_percentages*weights[mask],axis=0)
total_n_percentages =total_n_percentages/sum(total_n_percentages)*100

total_v_percentages = np.sum(total_v_percentages*weights[mask],axis=0)
total_v_percentages = total_v_percentages/sum(total_v_percentages)*100
# recalculated_npercentages = np.sum(total_v_percentages/ (bin_centers*1E-6)**3,axis=0)

# recalculated_npercentages = recalculated_npercentages/sum(recalculated_npercentages)*100


fig = plt.figure(figsize=(6,4))
# plt.grid(which='both', linestyle='--', linewidth=0.5)
# plt.bar(bin_edges[:-1], total_v_percentages, width=bin_widths, align='edge', edgecolor='black')

# # Add labels

# plt.ylabel("Volume PDF (%)")
# #ax1.set_title(f"t= {round(t_start*1000)} to {round(t_end*1000)} ms, \n T: {transmission:.1f} %, num. records: {num_records} ")
# plt.xscale('log')
# #plt.yscale('log')

# plt.ylim(1e-1,40)
# plt.xlim(bin_edges[0],bin_edges[-1])




plt.step(bin_edges[:-1], total_n_percentages, where="post", color=colors[1])

# Add labels
plt.xlabel(r"Diameter ($\mathrm{\mu}$m)")
plt.ylabel("Number distribution (%)")
#plt.title(f"Particle distribution at {date}, \n t= {round(t_start*1000)} to {round(t_end*1000)} ms, transmission: {transmission:.1f} % ")
plt.xscale('log')
#plt.yscale('log')
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.ylim(1e-1,20)
plt.xlim(1,500)
x= np.array(bin_edges[:-1],dtype='float')
y =np.array(total_n_percentages,dtype='float')
mask = np.isfinite(x) & np.isfinite(y)
x = x[mask]
y = y[mask]
plt.fill_between(x, 0, y, step='post', color=colors[1], alpha=0.2)
print(f"filename: {filename}")
full_save_path = os.path.join(save_path,filename)
print(f"full path: {full_save_path}")
plt.tight_layout()


plt.savefig(fr"C:\Users\sikke\Documents\universiteit\Master\Thesis\presentation\measurementseries0dot03_averagemorphing20.svg")
plt.show()
#plt.savefig(full_save_path+"_seriesaverage.svg")

###saving this set to plot in one frame
full_series_savepath = os.path.join(series_savepath,keyphrase)
bin_edges =np.array(bin_edges)
total_n_percentages = np.array(total_n_percentages)

bin_widths = np.array(bin_widths)
#np.savez(full_series_savepath,n_percentages=total_n_percentages,bins=bin_edges[:-1],bin_widths=bin_widths)