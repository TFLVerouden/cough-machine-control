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
        keep= [ "waterjet","waterjet_camera"] #"PEOjet",
    return keep

save_names= ["concentration", "film_thickness", "pressure", "height","jets"] #choose which one you want


save_name = "height"
save_name ="concentration"
keep = comparison(save_name)


filtered = [f for f in npz_files if any(k in f for k in keep)]
# colors = plt.get_cmap("tab10").colors   # tab10 = Tableau ColorBlind10
print(len(filtered))

plt.figure()
i=0



for file in filtered:
    filename = os.path.basename(file)   # "PEO_sample1_123.txt"
    print(filename)

    parts = filename.split("_")
    print(parts)
    if save_name!= "jets":
        label_fluid = parts[0]
        if label_fluid== "water":
            label_con  = ""
            label_amount = parts[1]
            label_cough= parts[2]
        else:
            label_con  = parts[1] 
            label_amount = parts[2]
            label_cough= parts[3]

        if label_fluid=="PEO":
            label_fluid = "PEO 2M"
        elif label_fluid=="PEO600K":
            label_fluid="PEO 600K"
    
        label_con = label_con.replace("dot", ".")
        label_con =label_con.replace("percent","")
        if label_fluid == "water":
            label_fluid = "Water"
        else:
            label_con = label_con + "% " 
        if save_name =="height":
            if "lower" in parts[2] or "higher" in parts[2]:
            
                label_end= parts[2]
                label_end = label_end.replace("cm","cm ")
                label_amount= label_cough
                label_cough = parts[4]
                label_cough = label_cough + " " + label_end
        label_amount = label_amount.replace("dot", ".")
        label_amount = label_amount.replace("ml", "mL")
        label_cough = label_cough.replace("dot", ".")
        full_label = label_fluid + " " + label_con  + label_amount + " " + label_cough
    else:
        full_label = filename.split(".")[0]
    data = np.load(file,allow_pickle=True)
    bins = data['bins']
    n_percentages = data['n_percentages']
    bin_widths= data['bin_widths']

    plt.step(bins,n_percentages,where="post",color=colors[i],label=full_label)
    plt.grid(which="both",axis='both',linestyle='--', linewidth=0.5)
    plt.ylim(0,50)
    plt.xscale('log')
    plt.xlabel(r"Diameter ($\mu$m)")
    plt.ylabel("Number distribution (%)")
    i+=1
plt.legend()

#plt.savefig(total_savepath+"\\" +save_name + "comparison.svg")
plt.show()