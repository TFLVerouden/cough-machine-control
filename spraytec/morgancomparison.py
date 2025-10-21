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
    if save_name == "Morgan":
        keep = ["PEO600K_0dot2_1ml_1dot5bar",
        "water_1ml_1dot5bar","Newtonian","0.2%wt"]
    elif save_name =="film_thickness":
        keep = ["PEO_0dot25_1ml_1dot5bar",
            "PEO_0dot25_1dot5ml_1dot5bar"]
    elif save_name == "pressure":
            keep = [
        "PEO_0dot25_1dot5ml_"]  # catches any pressure variant]
    if save_name== "height":
        keep = [
        "PEO_0dot25_1ml_1dot5bar_80ms.npz",
        "PEO_0dot25_2cmlower_1ml_1dot5bar_80ms.npz"]
    if save_name =="jets":
        keep= [ "waterjet","PEOjet"] #"PEOjet",
    return keep

save_names= ["concentration", "film_thickness", "pressure", "height","jets"] #choose which one you want


save_name = "Morgan"

keep = comparison(save_name)


filtered = [f for f in npz_files if any(k in f for k in keep)]
# colors = plt.get_cmap("tab10").colors   # tab10 = Tableau ColorBlind10
print(len(filtered))



i=0

labels = ["Li et al. 0.2% PEO600K","Li et al. Water", "Sikkema 0.2% PEO600K", "Sikkema Water", ]
text= ["a.","b."]
linestyles= ["--","-"]
plt.subplots(1,2,figsize= (6,4),sharex=True,sharey=True)
for file in filtered:
    filename = os.path.basename(file)   # "PEO_sample1_123.txt"
    label= labels[i]
    parts = filename.split("_")
    print(parts)
    plt.subplot(1,2,i%2+1)
    data = np.load(file,allow_pickle=True)
    bins = data['bins']
    n_percentages = data['n_percentages']
    
    bin_widths= data['bin_widths']
 
    average = np.average(bins, weights=n_percentages)
    # median, Q1, Q3 using np.quantile with weights
    def weighted_quantile(values, quantiles, sample_weight):
        values = np.array(values, dtype=float)
        sample_weight = np.array(sample_weight, dtype=float)
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]
        cdf = np.cumsum(sample_weight) / np.sum(sample_weight)
        return np.interp(quantiles, cdf, values)
    weights = n_percentages/100
    median = weighted_quantile(bins, 0.5, weights/100)
    q1 = weighted_quantile(bins, 0.25, weights/100)
    q3 = weighted_quantile(bins, 0.75, weights/100)
    print(f"{label} mean: {average:.2f},median:{median:.2f},LQ:{q1:.2f},UQ:{q3:.2f}")
    plt.step(bins,n_percentages,where="post",color=colors[i//2], linestyle= linestyles[i//2],label=label)
    plt.text(0.1, 23, text[i%2], 
             fontsize=14, fontweight='bold', va='top')
    plt.grid(which="both",axis='both',linestyle="--", linewidth=0.5)
    plt.ylim(0,25)
    plt.xscale('log')
    plt.xlabel(r"Diameter ($\mu$m)")
    plt.ylabel("Number distribution (%)")
    
    i+=1

plt.tight_layout()
plt.savefig(total_savepath+"\\" +save_name + "comparison.svg")
plt.show()