import numpy as np
import matplotlib.pyplot as plt
import os
import sys
#plt.style.use("tableau-colorblind10")
cwd = os.path.dirname(os.path.abspath(__file__))
print(cwd)
parent_dir = os.path.dirname(cwd)
function_dir = os.path.join(parent_dir,'functions')

sys.path.append(function_dir)
from cvd_check import set_cvd_friendly_colors

colors = set_cvd_friendly_colors()

save_path = os.path.join(cwd,"results_spraytec","Serie_Averages")
series_savepath = os.path.join(save_path,"npz_files")
total_savepath = os.path.join(save_path,"bundled")
os.makedirs(total_savepath, exist_ok=True)

npz_files = [os.path.join(series_savepath, f) for f in os.listdir(series_savepath) if f.endswith('.npz')]
#print(npz_files)

def comparison(save_name):
    if save_name == "concentration":
        keep = ["PEO600K_0dot2_1ml_1dot5bar", "PEO_0dot03_1dot5ml_1dot5bar",
                "PEO_0dot25_1dot5ml_1dot5bar", "PEO_1percent_1dot5ml_1dot5bar",
                "water_1ml_1dot5bar"]
    elif save_name =="film_thickness":
        keep = ["PEO_0dot25_1ml_1dot5bar", "PEO_0dot25_1dot5ml_1dot5bar"]
    elif save_name == "pressure":
        keep = ["PEO_0dot25_1dot5ml_"]  # any pressure variant
    elif save_name== "height":
        keep = ["PEO_0dot25_1ml_1dot5bar_80ms.npz", "PEO_0dot25_2cmlower_1ml_1dot5bar_80ms.npz"]
    elif save_name =="jets":
        keep = ["waterjet", "PEOjet"]
    return keep

# ---------------- Compute modes ----------------
save_name = "concentration"  # or "jets", "height", etc.
keep = comparison(save_name)

filtered = [f for f in npz_files if any(k in f for k in keep)]
print(f"Found {len(filtered)} files for {save_name}")

mode_results = {}  # file_name -> mode info

for file in filtered:
    filename = os.path.basename(file)
    data = np.load(file, allow_pickle=True)
    bins = data['bins']
    n_percentages = data['n_percentages']
    bin_widths = data['bin_widths']

    # Histogram-based mode
    max_index = np.argmax(n_percentages)
    mode_bin_left = bins[max_index]
    mode_bin_right = bins[max_index] + bin_widths[max_index]
    mode_bin_center = (mode_bin_left + mode_bin_right) / 2
    ###Plotting to show that the plotting goes right
    # plt.step(bins,n_percentages,where="post")
    # plt.vlines(mode_bin_left,0,20,color='r')
    # plt.xscale('log')
    # plt.show()
    mode_results[filename] = {
        "mode_center": mode_bin_center,
        "mode_left": mode_bin_left,
        "mode_right": mode_bin_right
    }

# ---------------- Print modes ----------------
for fname, m in mode_results.items():
    print(f"{fname}: center={m['mode_center']:.2f}, left={m['mode_left']:.2f}, right={m['mode_right']:.2f}")

# ---------------- Save modes ----------------
np.savez(r"C:\Users\sikke\Documents\GitHub\cough-machine-control\other_side_stuff\modedropletsizes\mode_results_Abe.npz", **mode_results)


print("Done! Modes saved.")