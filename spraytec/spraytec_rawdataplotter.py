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

#we use this fur future compatibility
pd.set_option('future.no_silent_downcasting', True)

#nice style
matplotlib.use("TkAgg")  # Or "Agg", "Qt5Agg", "QtAgg"
plt.rcParams.update({'font.size': 14})

#FINDING THE FILES
cwd = os.path.dirname(os.path.abspath(__file__))

path = os.path.join(cwd,"individual_data_files")
save_path = os.path.join(cwd,"results_spraytec","time_plots")




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
for file in matching_files:

    filename = file.split('\\')[-1].replace('.txt', '')
 
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
    t_end = df_filtered.loc[time_chosen,"Time (relative)"]
    t_start = t_end - df_filtered.loc[time_chosen,"Duration"]
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

    # plt.figure(figsize=(9,6))
    # plt.bar(bin_edges[:-1], percentages, width=bin_widths, align='edge', edgecolor='black')

    # # Add labels
    # plt.xlabel(r"Diameter ($\mu$m)")
    # plt.ylabel("Volume Percentage (%)")
    # plt.title(f"Particle distribution at {date}, \n t= {round(t_start*1000)} to {round(t_end*1000)} ms, transmission: {transmission:.1f} % ")
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.grid(which='both', linestyle='--', linewidth=0.5)
    # plt.ylim(1e-1,40)
    # plt.xlim(bin_edges[0],bin_edges[-1])
    # plt.show()


    #NUMBER PERCENTAGES
    n_percentages = percentages/ (bin_centers*1E-6)**3
    n_percentages  = n_percentages/ sum(n_percentages)*100
    # plt.figure(figsize=(9,6))
    # plt.bar(bin_edges[:-1], n_percentages, width=bin_widths, align='edge', edgecolor='black')

    # # Add labels
    # plt.xlabel(r"Diameter ($\mu$m)")
    # plt.ylabel("Number Percentage (%)")
    # plt.title(f"Particle distribution at {date}, \n t= {round(t_start*1000)} to {round(t_end*1000)} ms, transmission: {transmission:.1f} % ")
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.grid(which='both', linestyle='--', linewidth=0.5)
    # plt.ylim(1e-1,40)
    # plt.xlim(bin_edges[0],bin_edges[-1])
    # # plt.show()

    #Over time plot
    len_arr= df_filtered.shape[0]
    dates = np.array([],dtype=str)
    percentages_all = np.zeros((len(columns_scattervalues),len_arr)) ### (diameter_values,time)

    times = np.zeros(len_arr)
    transmissions = np.zeros(len_arr)
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
        t_end = df_filtered.loc[i,"Time (relative)"]
        
        t_start= t_end - df_filtered.loc[i,"Duration"]
        times[i] = t_start
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
    n_percentages = percentages_all/ (bin_centers.reshape(-1,1)*1E-6)**3
    n_percentages  = n_percentages/ np.sum(n_percentages,axis=0)*100
    # Make meshgrid of edges
    X, Y = np.meshgrid(time_edges, diameter_edges)

    vmin = 0
    vmax = 40
    plt.figure()
    pcm = plt.pcolormesh(X, Y, n_percentages,
                        norm=LogNorm(vmin=1e-1, vmax=5e1),
                        cmap= 'Blues')
    cbar2 = plt.colorbar(pcm)

    cbar2.set_label('Number distribution (%)')
    plt.yscale('log')
    plt.grid(which='major',linestyle="--",alpha=0.8)
    plt.ylabel(r'D ($\mu$m)')
    plt.xlim(0,0.2)
    plt.ylim(bin_centers[0], bin_centers[-1])
    plt.xlabel('Time (s)')
    full_save_path = os.path.join(save_path,filename)
    plt.savefig(full_save_path+".svg")
    
    

    #fig, ax = plt.subplots(2,1,sharex=True,sharey=True,figsize=(9,6))
    #fig = plt.figure(figsize=(9, 6))

   
    #gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 0.3])  # Smaller 3rd row
    # gs = GridSpec(3, 2, width_ratios=[20, 1], height_ratios=[1, 1, 0.3], figure=fig)

    # ax0 = fig.add_subplot(gs[0, 0])
    # ax1 = fig.add_subplot(gs[1, 0],sharex=ax0,sharey=ax0)
    # ax2 = fig.add_subplot(gs[2, 0],sharex=ax0)  # Span both columns
    # cax0 = fig.add_subplot(gs[0, 1])
    # cax1 = fig.add_subplot(gs[1, 1])
    # # cax0 = inset_axes(ax0, width="2%", height="100%", loc="right", borderpad=-1)
    # # cax1 = inset_axes(ax1, width="2%", height="100%", loc="right", borderpad=-1)


    # plt.setp(ax0.get_xticklabels(), visible=False)
    # plt.setp(ax1.get_xticklabels(), visible=False)
    # pcm = ax0.pcolormesh(X, Y, percentages_all,
    #                     norm=LogNorm(vmin=1e-1, vmax=5e1),
    #                     cmap= 'Blues')

    # ax0.set_yscale('log')
    # # ax0.set_xlabel('Time (s)')
    # ax0.set_ylabel(r'D ($\mu$m)')
    # ax0.set_xlim(0,0.2)
    # ax0.set_ylim(bin_centers[0], bin_centers[-1])

    # ax0.grid(which='major',linestyle="--",alpha=0.8)
    # # ax2 = ax.twinx()
    # # ax2.plot(times,100-transmissions,c="r")
    # # ax2.set_ylim(0,100)
    # # ax2.set_ylabel("Reflected (%)")
    # cbar = plt.colorbar(pcm, cax=cax0)

    # cbar.set_label('PDF (Volume %)')
    # # pos = cbar.ax.get_position()  # get current position [x0, y0, width, height]

    # # # Move the colorbar right by increasing x0 and x1:
    # # new_pos = [pos.x0 + 0.05, pos.y0, pos.width, pos.height]
    # # cbar.ax.set_position(new_pos)

    # ### Number percentage

    # pcm2 = ax1.pcolormesh(X, Y, n_percentages,
    #                     norm=LogNorm(vmin=1e-1, vmax=5e1),
    #                     cmap= 'Blues')

    # ax1.set_yscale('log')
    # ax1.grid(which='major',linestyle="--",alpha=0.8)
    # ax1.set_ylabel(r'D ($\mu$m)')
    # ax1.set_xlim(0,0.2)
    # ax1.set_ylim(bin_centers[0], bin_centers[-1])


    # # ax2 = ax.twinx()
    # # ax2.plot(times,100-transmissions,c="r")
    # # ax2.set_ylim(0,100)
    # # ax2.set_ylabel("Reflected (%)")

    # cbar2 = plt.colorbar(pcm2, cax=cax1)

    # cbar2.set_label('PDF (Number %)')
    # # pos = cbar2.ax.get_position()  # get current position [x0, y0, width, height]

    # # Move the colorbar right by increasing x0 and x1:
    # # new_pos = [pos.x0 + 0.05, pos.y0, pos.width, pos.height]
    # # cbar2.ax.set_position(new_pos)

    # ###Transmission
    # ax2.plot(times,transmissions,c="r")
    # ax2.set_ylim(60,100)
    # ax2.set_xlim(0,0.2)
    # ax2.set_ylabel("T (%)")
    # ax2.set_aspect('auto') 
    # ax2.set_xlabel('Time (s)')


    # full_save_path = os.path.join(save_path,filename)
    # plt.savefig(full_save_path+".svg")
    plt.close('all')
    #plt.show()

