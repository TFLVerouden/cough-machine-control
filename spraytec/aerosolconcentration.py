"""
Shows first entry the graph and over time an imshow of the spraytec data
"""
keyphrase = "PEO_1percent"  ##change this for different statistics
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
import sys
cwd = os.path.dirname(os.path.abspath(__file__))
print(cwd)
parent_dir = os.path.dirname(cwd)
function_dir = os.path.join(parent_dir,'functions')

sys.path.append(function_dir)
from cvd_check import set_cvd_friendly_colors

colors = set_cvd_friendly_colors()
#we use this fur future compatibility
pd.set_option('future.no_silent_downcasting', True)

#nice style
matplotlib.use("TkAgg")  # Or "Agg", "Qt5Agg", "QtAgg"
plt.rcParams.update({'font.size': 14})

#FINDING THE FILES
cwd = os.path.dirname(os.path.abspath(__file__))

path = os.path.join(cwd,"individual_data_files")
save_path = os.path.join(cwd,"results_spraytec","aerosol_concentration")




txt_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')]

pattern = re.compile(rf"\d{{4}}_\d{{2}}_\d{{2}}_\d{{2}}_\d{{2}}_{re.escape(keyphrase)}_\d+(?:_.*)?\.txt")

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
all_cv= []

plt.figure()
j=0
for file in matching_files:
    
    if j<6: 
        linestyle= "-"
        markerstyle= "o"
    else: 
        linestyle= "-"
        markerstyle= "x"
    j+=1
    filename = file.split('\\')[-1].replace('.txt', '')
 
    #From here we read the data

    df = pd.read_table(file,delimiter=",", encoding="latin-1")
    df = df.replace('-', 0)
    print(df.loc[0,"Date-Time"])
    for col in df.columns:
        # Try converting each column to numeric, coercing errors to NaN
        df[col] = pd.to_numeric(df[col], errors='ignore')
    important_columns= ["Date-Time","Transmission", "Duration","Time (relative)","Cv(%)"]

    columns_scattervalues = df.loc[:,"0.10000020":"1000.00195313"].columns.tolist()
    #df[columns_scattervalues].astype(float)
    important_columns = important_columns + columns_scattervalues
    df_filtered= df.loc[:,important_columns]

    ####TEST

    #time depended variables
    time_chosen = 1
    num_record =df.loc[:,"Number of records in average "].sum()

    date= df_filtered.loc[time_chosen,"Date-Time"]
    percentages = df_filtered.loc[time_chosen,columns_scattervalues]
    t_end = df_filtered.loc[time_chosen,"Time (relative)"]
    t_start = t_end - df_filtered.loc[time_chosen,"Duration"]
    transmission = df_filtered.loc[time_chosen,"Transmission"]
    cv = df_filtered.loc[:,"Cv(%)"]
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

    #Over time plot
    len_arr= df_filtered.shape[0]
    cv_all = np.zeros(len_arr) ### (diameter_values,time)

    times = np.zeros(len_arr)
    
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
    t=  df_filtered.loc[:,"Time (relative)"] - df_filtered.loc[:,"Duration"]
    #print(df_filtered["Time (relative)"].diff().iloc[1:])
    #print(df_filtered.iloc[:-1]["Duration"])
    print(cv.mean())
    #plt.scatter(df_filtered["Time (relative)"].diff().iloc[1:] ,df_filtered.iloc[:-1]["Duration"])
    #plt.show()
    meancv = cv.mean()
    plt.plot(t,cv,linestyle=linestyle,marker= markerstyle,label=f"avg cv: {meancv:.3f}, no. rec:{num_record}")
plt.ylabel("Cv (%)")
plt.xlabel("Time (s)")
plt.legend()
plt.show()
   

