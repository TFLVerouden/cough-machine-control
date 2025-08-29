import numpy as np
import matplotlib.pyplot as plt
import os
plt.style.use("tableau-colorblind10")
cwd = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(cwd,"results_spraytec","Serie_Averages")
series_savepath = os.path.join(save_path,"npz_files")
total_savepath = os.path.join(save_path,"bundled")
os.makedirs(total_savepath, exist_ok=True)

npz_files = [os.path.join(series_savepath, f) for f in os.listdir(series_savepath) if f.endswith('.npz')]
print(npz_files)
colors = plt.get_cmap("tab10").colors   # tab10 = Tableau ColorBlind10

plt.figure()
i=0
for file in npz_files:
    filename = os.path.basename(file)   # "PEO_sample1_123.txt"


    parts = filename.split("_")
    label  = parts[1]
    label = label.replace("dot", ".")
    label =label.replace("percent","")
   
    data = np.load(file,allow_pickle=True)

    print(label)
    bins = data['bins']
    n_percentages = data['n_percentages']
    bin_widths= data['bin_widths']
    print(bins)
    print(n_percentages)

    plt.step(bins,n_percentages,where="post",color=colors[i],label="PEO 2M " + label + "%")
    plt.grid(which="both",axis='both',linestyle='--', linewidth=0.5)
    plt.ylim(0,25)
    plt.xscale('log')
    plt.xlabel(r"Diameter ($\mu$m)")
    plt.ylabel("Number distribution (%)")
    i+=1
plt.legend()

plt.savefig(total_savepath+"\\comparison.pdf")
plt.show()