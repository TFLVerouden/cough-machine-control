import os
import numpy as np
import h5py
import pickle
import matplotlib.pyplot as plt
# -------------------- Settings --------------------
casenames = ["050B_water", "050B_0pt05wt", "050B_0pt1wt", "050B_0pt2wt"]

cwd = os.path.abspath(os.path.dirname(__file__))
FilePath = os.path.join(cwd, "PDA")

# Spraytec bins file
spraytec_file = "spraytec/results_spraytec/Serie_Averages/npz_files/water_1ml_1dot5bar_80ms.npz"
data = np.load(spraytec_file, allow_pickle=True)
bins = data['bins']
bin_widths = data['bin_widths']
d_edges= np.append(bins,bins[-1]+bin_widths[-1])
d_bins =(d_edges[:-1] + d_edges[1:]) / 2


# -------------------- Extract modes --------------------
mode_results = {}

for casename in casenames:
    print(f"Processing case: {casename}")
    folder = os.path.join(FilePath, casename)
    FileList = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".h5")]

    case_modes = []
    d_all = []

    PDA_case = []
    for fname in FileList:
        with h5py.File(fname, "r") as f:
            d_vals = f["/BSA/Diameter"][:]
            t = f["/BSA/Arrival_Time"][:]
        
        PDA_case.append({
            "fname": os.path.basename(fname),
            "d": d_vals,
            "t": t
        })

        # Aggregate all, early, late
        d_all.extend(d_vals)
        

    d= np.array(d_all)    # Histogram

    ### the morgan way to define the bins
    # log_d_edges = np.arange(0, 4.8 + 0.3, 0.3)  # log-space edges of Morgan
    # #log_d_edges = np.arange(0, 4.8 + 0.2, 0.2)
    # d_edges = np.exp(log_d_edges) #converted to linear space
    # # Compute PDFs
    # dx = np.diff(d_edges) #Bin_widths in linear space
    # bin_widths = dx
    # d_bins = (d_edges[:-1] + d_edges[1:]) / 2

    ###
    print(len(d))
    
    pdf_all, _ = np.histogram(d, bins=d_edges)
    
    n_pdf = pdf_all / np.sum(pdf_all * bin_widths)
    plt.step(d_bins,n_pdf/sum(n_pdf)*100)
    plt.title(casename)
    plt.xscale('log')
    plt.show()
    # Mode bin center
    max_bin_index = np.argmax(n_pdf)
    mode_bin_left = d_edges[max_bin_index]
    mode_bin_right = d_edges[max_bin_index + 1]
    mode_bin_center = (mode_bin_left + mode_bin_right) / 2

    case_modes.append({
            "mode_center": mode_bin_center,
            "mode_left": mode_bin_left,
            "mode_right": mode_bin_right,
        })
    mode_results[casename] = case_modes

# -------------------- Save results --------------------
# Save as npz
np.savez(r"C:\Users\sikke\Documents\GitHub\cough-machine-control\other_side_stuff\modedropletsizes\mode_results_Morgan.npz", **mode_results)

# If you just created mode_results in memory
for casename, modes in mode_results.items():
    print(f"{casename}:")
    for i, m in enumerate(modes, start=1):
        print(f"  File {i}: center={m['mode_center']:.2f}, left={m['mode_left']:.2f}, right={m['mode_right']:.2f}")


print("Done! Saved mode values for all casenames.")
print("Keys in results:", list(mode_results.keys()))
