"""
This file finds all number and volume means,stds and skewnesses for a given keyphrase and makes a csv of these values
"""
keyphrase = "PEO_1percent_1dot5ml_1dot5bar_80ms"  ##change this for different statistics

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
import csv

#we use this fur future compatibility
pd.set_option('future.no_silent_downcasting', True)

#nice style
matplotlib.use("TkAgg")  # Or "Agg", "Qt5Agg", "QtAgg"
plt.rcParams.update({'font.size': 14})

#FINDING THE FILES
cwd = os.path.dirname(os.path.abspath(__file__))

path = os.path.join(cwd,"Averages")
path = os.path.join(path,"Unweighted","1percent") #for the unweighted ones

save_path = os.path.join(cwd,"results_spraytec","csv")
txt_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')]

v_means = []
v_stds = []
v_skewnesses = []
n_means = []
n_stds = []
n_skewnesses = []
t_starts =[]
t_ends =[]
num_records =[]


pattern = re.compile(rf"average_{re.escape(keyphrase)}_\d+(?:_.*)?\.txt")

# Filter matching files
matching_files = [f for f in txt_files if pattern.search(os.path.basename(f))]

for file in matching_files:

    filename = file.split('\\')[-1].replace('.txt', '')
    
    #From here we read the data
    
    
    df = pd.read_table(file,delimiter=",",encoding='latin1')
    df = df.replace('-', 0)
    for col in df.columns:
        # Try converting each column to numeric, coercing errors to NaN
        df[col] = pd.to_numeric(df[col], errors='ignore')

    columns_scattervalues = df.loc[:,"% V (0.100-0.117µm)":"% V (857.698-1000.002µm)"].columns.tolist()


    t_end = df.loc[0,"Time (relative)"]    
    t_start= t_end - df.loc[0,"Duration"]
    num_record =df.loc[0,"Number of records in average "]
    percentages = df.loc[0,columns_scattervalues]
    bin_centers = np.array([])
    for column in columns_scattervalues:
        match = re.search(r"\(([\d.]+)-([\d.]+)", column)
        if match:
            lower = float(match.group(1))
            upper = float(match.group(2))
            center = (lower + upper) / 2
            bin_centers = np.append(bin_centers,center)
    

    n_percentages = percentages/ (bin_centers*1E-6)**3
    n_percentages  = n_percentages/ sum(n_percentages)*100
    mean = np.sum(bin_centers * percentages/100 )
    variance = np.sum(percentages/100 * (bin_centers - mean)**2)
    std = np.sqrt(variance)
    skewness = np.sum(percentages * (bin_centers - mean)**3) / (std**3)
    number_mean = np.sum(bin_centers * n_percentages/100 )
    number_variance = np.sum(n_percentages/100 * (bin_centers - number_mean)**2)
    number_std = np.sqrt(number_variance)
    number_skewness = np.sum(n_percentages * (bin_centers - number_mean)**3) / (number_std**3)

    v_means.append(mean)
    v_stds.append(std)
    v_skewnesses.append(skewness)
    n_means.append(number_mean)
    n_stds.append(number_std)
    n_skewnesses.append(number_skewness)
    t_starts.append(t_start)
    t_ends.append(t_end)
    num_records.append(num_record)

    # print(f"Volume,mean: {mean:.2f},std: {std:.2f},skewness: {skewness}")
    # print(f"Number: mean: {number_mean:.2f},std: {number_std:.2f}, skewness: {number_skewness}")

stats = {
    "t_start" : t_starts,
    "t_ends" : t_ends,
    "num_records": num_records,
    "v_means": v_means,
    "v_stds": v_stds,
    "v_skewnesses": v_skewnesses,
    "n_means": n_means,
    "n_stds": n_stds,
    "n_skewnesses": n_skewnesses
}

full_save_path = os.path.join(save_path, keyphrase + ".csv")
print(f"saved at: {full_save_path}")
num_rows = len(v_means)
with open(full_save_path, 'w', newline='') as f:
    writer = csv.writer(f)
    
    # Header with filename column first
    header = ["filename", "Index"] + list(stats.keys())
    writer.writerow(header)
    
    for i in range(num_rows):
        row = [keyphrase, i] + [stats[key][i] for key in stats]
        writer.writerow(row)
