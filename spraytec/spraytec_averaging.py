"""
Produces the average plots of the spraytec data either via a loop over a keyphrase or via a file explorer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import tkinter as tk
from tkinter import filedialog
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import re

#we use this fur future compatibility
pd.set_option('future.no_silent_downcasting', True)

#nice style
matplotlib.use("TkAgg")  # Or "Agg", "Qt5Agg", "QtAgg"
plt.rcParams.update({'font.size': 14})

#FINDING THE FILES
cwd = os.path.dirname(os.path.abspath(__file__))

path = os.path.join(cwd,"Averages")
print(f"Path: {path}")
save_path = os.path.join(cwd,"results_spraytec","Averages")
print(f"Save path {save_path}")
keyphrase = "PEO_0dot03_1dot5ml_1dot5bar_80ms"  ##change this for different statistics

txt_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')]
pattern = re.compile(rf"average_{re.escape(keyphrase)}_\d+\.txt")

# Filter matching files
matching_files = [f for f in txt_files if pattern.search(os.path.basename(f))]
save_path = os.path.join(save_path,keyphrase)

# Create folder if it doesn't exist
os.makedirs(save_path, exist_ok=True)


# #NOW WE CHOSE with a dialog!!!!
# root = tk.Tk()
# root.withdraw()  # hide the root window

# # Let user pick a file inside your directory
# file = filedialog.askopenfilename(initialdir=path, filetypes=[("Text files", "*.txt")])

# print("You picked:", file)
#### The loop over all files
for file in matching_files:

    filename = file.split('\\')[-1].replace('.txt', '')

    #From here we read the data

    df = pd.read_table(file,delimiter=",",encoding='latin1')
    df = df.replace('-', 0)

    print(df.loc[0,"Date-Time"])
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

    ### VOLUME PERCENTAGES

    fig,(ax1,ax2) = plt.subplots(2,1,sharex=True,sharey=True)
    ####OPTINAL CUMSUM INClUSION, change (ax1,ax2 to ax)
    #cdf_v = np.cumsum(percentages)
    #ax[0].plot(bin_centers,cdf_v,c='r')
    #ax[0].set_ylabel("Volume CDF (%)")

    #ax1 = plt.twinx(ax=ax[0])
    #### ENDS HERE
    ax1.grid(which='both', linestyle='--', linewidth=0.5)
    ax1.bar(bin_edges[:-1], percentages, width=bin_widths, align='edge', edgecolor='black')

    # Add labels

    ax1.set_ylabel("Volume PDF (%)")
    ax1.set_title(f"t= {round(t_start*1000)} to {round(t_end*1000)} ms, \n T: {transmission:.1f} %, num. records: {num_records} ")
    ax1.set_xscale('log')
    #plt.yscale('log')

    ax1.set_ylim(1e-1,40)
    ax1.set_xlim(bin_edges[0],bin_edges[-1])



    #NUMBER PERCENTAGES
    n_percentages = percentages/ (bin_centers*1E-6)**3
    n_percentages  = n_percentages/ sum(n_percentages)*100

    ####OPTINAL CUMSUM INClUSION
    # cdf_n = np.cumsum(n_percentages)

    # ax[1].grid(which='both', linestyle='--', linewidth=0.5)

    # ax[1].plot(bin_centers,cdf_n,c='r')
    # ax[1].set_ylim(0,100)
    # ax[1].set_ylabel("Number CDF (%)")
    # ax2= plt.twinx(ax=ax[1])
    #ENDS HERE
    ax2.bar(bin_edges[:-1], n_percentages, width=bin_widths, align='edge', edgecolor='black')

    # Add labels
    ax2.set_xlabel(r"Diameter ($\mu$m)")
    ax2.set_ylabel("Number PDF (%)")
    #plt.title(f"Particle distribution at {date}, \n t= {round(t_start*1000)} to {round(t_end*1000)} ms, transmission: {transmission:.1f} % ")
    ax2.set_xscale('log')
    #plt.yscale('log')
    ax2.grid(which='both', linestyle='--', linewidth=0.5)
    ax2.set_ylim(1e-1,40)
    ax2.set_xlim(bin_edges[0],bin_edges[-1])
    print(f"filename: {filename}")
    full_save_path = os.path.join(save_path,filename)
    print(f"full path: {full_save_path}")
    
    plt.savefig(full_save_path+".svg")

    
