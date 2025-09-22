import os
"""
Produces the average plots of the spraytec data either via a loop over a keyphrase or via a file explorer
"""
keyphrase = "PEO600K_0dot2_1ml_1dot5bar_80ms"  ##change this for different statistics

#keyphrase = "waterjet"  ##change this for different statistics

cwd = os.path.dirname(os.path.abspath(__file__))

path = os.path.join(cwd,"Averages")
path = os.path.join(path,"Unweighted","600k_0dot2") #for the unweighted ones
#path = os.path.join(path,"weighted") #for the weighted ones

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
from scipy.stats import lognorm


#we use this fur future compatibility
pd.set_option('future.no_silent_downcasting', True)

#nice style
matplotlib.use("TkAgg")  # Or "Agg", "Qt5Agg", "QtAgg"
plt.rcParams.update({'font.size': 14})

#FINDING THE FILES

save_path = os.path.join(cwd,"results_spraytec","Averages")
print(f"Save path {save_path}")


txt_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')]
pattern = re.compile(rf"average_{re.escape(keyphrase)}_\d+(?:_.*)?\.txt")

# Filter matching files
matching_files = [f for f in txt_files if pattern.search(os.path.basename(f))]
save_path = os.path.join(save_path,keyphrase)

# Create folder if it doesn't exist
os.makedirs(save_path, exist_ok=True)

def lognormal_from_histogram(bin_centers, bin_edges, bin_widths, percentages):
    """
    Fit a log-normal distribution to a histogram and return the PDF at bin centers.

    Parameters:
        bin_centers (array-like): Centers of bins (linear scale)
        bin_edges (array-like): Edges of bins (linear scale)
        bin_widths (array-like): Widths of bins
        percentages (array-like): Percentage/counts per bin (sum does not need to be 1)

    Returns:
        mu (float): Mean in log-space
        sigma (float): Std dev in log-space
        pdf_values (np.array): Log-normal PDF evaluated at bin_centers
        lognorm_dist (scipy.stats._distn_infrastructure.rv_frozen): Fitted lognorm object
    """
    bin_centers = np.array(bin_centers, dtype=float)
    percentages = np.array(percentages, dtype=float)
    
    # Convert percentages to probabilities
    probabilities = percentages / np.sum(percentages)
    
    # Compute weighted log-space mean and std
    log_centers = np.log(bin_centers)
    mu = np.sum(probabilities * log_centers)
    sigma = np.sqrt(np.sum(probabilities * (log_centers - mu)**2))
    
    # Fit the lognormal distribution with fixed parameters
    lognorm_dist = lognorm(s=sigma, scale=np.exp(mu))
    
    # Evaluate PDF at bin centers
    pdf_values = lognorm_dist.pdf(bin_centers) 
    pdf_values = pdf_values/np.sum(pdf_values)*100
    mean = np.exp(mu + sigma**2 / 2)
    std = np.sqrt((np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2))
    return mean, std, pdf_values, lognorm_dist
# #NOW WE CHOSE with a dialog!!!!
# root = tk.Tk()
# root.withdraw()  # hide the root window

# # Let user pick a file inside your directory
# file = filedialog.askopenfilename(initialdir=path, filetypes=[("Text files", "*.txt")])

# print("You picked:", file)
#### The loop over all files
fig= plt.figure(figsize= (6,4))
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



    #NUMBER PERCENTAGES
    n_percentages = percentages/ (bin_centers*1E-6)**3
    n_percentages  = n_percentages/ sum(n_percentages)*100
    min_records =5
    max_records = 200
    weights = num_records
    max_n_check = np.max(n_percentages)
    max_n_limit = 75
    mask = (weights> min_records) & (weights<max_records) & (max_n_check<max_n_limit)
   

    ####OPTINAL CUMSUM INClUSION
    # cdf_n = np.cumsum(n_percentages)

    # ax[1].grid(which='both', linestyle='--', linewidth=0.5)

    # ax[1].plot(bin_centers,cdf_n,c='r')
    # ax[1].set_ylim(0,100)
    # ax[1].set_ylabel("Number CDF (%)")
    # ax2= plt.twinx(ax=ax[1])
    #ENDS HERE
    if mask:
        mean, std, pdf_values, lognorm_dist = lognormal_from_histogram(
        bin_centers, bin_edges, bin_widths, n_percentages)
        #print(pdf_values)
       
        #plt.bar(bin_edges[:-1], n_percentages, width=bin_widths, align='edge', edgecolor='black')
        plt.plot(bin_centers, pdf_values, "--",
                label=f'fit: mean={mean:.2f} μm, std={std:.2f} μm')

    # Add labels
plt.xlabel(r"Diameter ($\mu$m)")
plt.ylabel("Number PDF (%)")
plt.title(f"t= {round(t_start*1000)} to {round(t_end*1000)} ms, \n T: {transmission:.1f} %, num. records: {num_records} ")
plt.xscale('log')
#plt.yscale('log')
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.ylim(1e-1,40)
plt.xlim(bin_edges[0],bin_edges[-1])
print(f"filename: {filename}")
full_save_path = os.path.join(save_path,filename)
print(f"full path: {full_save_path}")
plt.tight_layout()
plt.legend()
#plt.savefig(full_save_path+".svg")
plt.show()
    
