import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 
import sys
cwd = os.path.dirname(os.path.abspath(__file__))
print(cwd)
parent_dir = os.path.dirname(cwd)
function_dir = os.path.join(parent_dir,'functions')

sys.path.append(function_dir)
from cvd_check import set_cvd_friendly_colors

colors = set_cvd_friendly_colors()

folder = r"C:\Users\sikke\Documents\GitHub\cough-machine-control\other_side_stuff\coughplotter"

csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]
plt.figure(figsize=(4,3))
plt.xlabel("Time (s)")
plt.ylabel("Flow rate (L/s)")

plt.xlim(0,0.5)
plt.ylim(0,11.3*5)
plt.grid(which="both")
i =0
for csv_file in csv_files:
    
    if i<6: 
        linestyle= "-"
    else: linestyle= "--"
    path  = os.path.join(folder,csv_file)
    name = csv_file.split("_")[1]
    name =name.replace(".csv","")
    reverse = csv_file.split("_")[-1]
    color = colors[i%6]
    if name== "Flowmeter":
        
        df = pd.read_csv(path)
        mask = df["Y"]>0
        df = df[mask]

    else:
        continue
        df = pd.read_csv(path,index_col=0)
    
    
    
    X,Y = df['X'], df['Y']
    x_0 = X.iloc[0]
    y_0 = Y.iloc[0]
    X -= x_0
    Y -=y_0
    if reverse != "r.csv":
        Y = -Y
    print(X,Y)

    
    plt.plot(X,Y*5,label= "Flow meter",linestyle=linestyle,marker="o",color="r")


   

    i+=1
npz_files = [f for f in os.listdir(folder) if f.endswith(".npz")]
j=0
# for npz_file in npz_files:

#     linestyle ="--"
#     path  = os.path.join(folder,npz_file)
#     name = npz_file.split(".")[0]
#     name =name.replace("_"," ")
#     name = name.replace("dot",".")
#     data = np.load(path)
#     if name == "PIV 1bar":
#         continue
    
#     flow_rate = data["flow_rate_Lps"]*1000
#     time = data["time_s"]
#     # --- Step 1: Define block size ---
#     block_size = 100

#     # --- Step 2: Average in blocks ---
#     n_blocks = len(flow_rate) // block_size  # full blocks only
#     flow_avg = flow_rate[:n_blocks*block_size].reshape(n_blocks, block_size).mean(axis=1)
#     time_avg = time[:n_blocks*block_size].reshape(n_blocks, block_size).mean(axis=1)

#     plt.plot(time_avg,flow_avg,color="k",linestyle=linestyle,label="PIV")
#     j+=1
#     i+=1

#plt.savefig(folder+"allcoughsliterature.svg") 

plt.grid(which="both")
plt.xlabel("Time (s)")
plt.ylabel("Flow velocity (m/s)")
plt.hlines(31.93,0,0.3,linestyles="--",label= "Velocity based on wad experiment")
plt.xlim(0,0.3)
plt.ylim(0)

plt.legend(loc='upper right')

#plt.savefig(folder+"allcoughsliterature.svg") 
plt.grid(which="both")
plt.tight_layout()
plt.savefig(rf"C:\Users\sikke\Documents\universiteit\Master\Thesis\presentation\PIVvswad.svg")
plt.show()