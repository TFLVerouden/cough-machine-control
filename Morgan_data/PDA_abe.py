import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from scipy import stats
from scipy.optimize import curve_fit
import sys
cwd = os.path.dirname(os.path.abspath(__file__))
print(cwd)
parent_dir = os.path.dirname(cwd)
function_dir = os.path.join(parent_dir,'functions')

sys.path.append(function_dir)
from cvd_check import set_cvd_friendly_colors

colors = set_cvd_friendly_colors()

def log_normal(bin_centers, mu, sigma):
    return 1/ (bin_centers *sigma * np.sqrt(2*np.pi)) *np.exp(-0.5* ((np.log(bin_centers)-mu)/sigma)**2)

# -------------------- Functions --------------------
def log_normal_fit(d, bin_edges):
    d = np.array(d)

    # Bin centers
    x_fit = (bin_edges[:-1] + bin_edges[1:]) / 2  #bin_centers

    shape, loc, scale = stats.lognorm.fit(d, floc=0)  # fix loc=0 if data > 0, lognorm fit scipy
    pdf_fit = stats.lognorm.pdf(x_fit, shape, loc=loc, scale=scale) #the real fit

    mu = np.log(scale)   # underlying normal mean
    sigma = shape        # underlying normal std
 
    mean = np.exp(mu + sigma**2 / 2)
    var = (np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2)
    std = np.sqrt(var)

    # Normalize to histogram area
    dx = np.diff(bin_edges) #bin_widths in linear space
    pdf_fit = pdf_fit / np.sum(pdf_fit*dx) #normalize like a pdf

    return x_fit, pdf_fit, mu, sigma

# -------------------- Settings --------------------
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 14})
cwd = os.path.abspath(os.path.dirname(__file__))

FilePath = os.path.join(cwd, "PDA")
savepath  = os.path.join(FilePath,"results")
os.makedirs(savepath, exist_ok=True)

casenames = ["050B_0pt2wt", "050B_water"]
legend_labels = ["0.2%wt", "Newtonian"]





# -------------------- Load Data --------------------
PDA_all = []
stats_all = []

for casename in casenames:
    folder = os.path.join(FilePath, casename)
    FileList = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".h5")]

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


    # Convert to arrays
    d_all = np.array(d_all)

    # plt.figure()
    # plt.plot(d_all,".")
    # plt.show()
    
    # Histogram bins
    log_d_edges = np.arange(0, 4.8 + 0.3, 0.3)  # log-space edges of Morgan
    
    d_edges = np.exp(log_d_edges) #converted to linear space
    # Compute PDFs
    dx = np.diff(d_edges) #Bin_widths in linear space
    d_bins = (d_edges[:-1] + d_edges[1:]) / 2 #bin_centers
 
    pdf_all,edges = np.histogram(d_all, bins=d_edges)#[0] #histogram based on the edges of Morgan

    
    pdf_all = pdf_all / np.sum(pdf_all*dx) #normalizing like a PDF
    

    stats_all.append({
        "d_all": d_all,
        "pdf_all": pdf_all,
    })
    PDA_all.append(PDA_case)

# -------------------- Plot Function --------------------
def plot_hist_and_fit(stats_all, label_list, pdf_type="all", ylim=(0, 0.13)):
    plt.figure(figsize=(6,4))
    for cc, stat in enumerate(stats_all):
        pdf = stat[f'pdf_{pdf_type}']
        
        plt.step(d_bins, pdf, where='mid',color=colors[cc]) # plotting the histogram as bin_centers in the middle
        x_fit, pdf_fit, mu_d, sigma_d = log_normal_fit(stat[f'd_{pdf_type}'], d_edges) #fitting
        popt, pcov = curve_fit(log_normal, d_bins, pdf)
        plt.plot(d_bins, log_normal(d_bins, *popt), '--',color= colors[cc],
         label='fit: mu=%5.3f, sigma=%5.3f' % tuple(popt))
        plt.plot(x_fit, pdf_fit, ':',color = colors[cc],
                 label=f'Calculated {label_list[cc]}, \n mu={mu_d:.1f} μm, \n std={sigma_d:.1f} μm') #plotting the fit
    plt.title(f"{pdf_type}")
    plt.xscale("log")
    plt.xlabel(r"$d$ [$\mu$m]")
    plt.ylabel(r"p.d.f. [$\mu$m$^{-1}$]")
    #plt.ylim(*ylim)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=14, frameon=False)
    plt.tight_layout()
    plt.savefig(savepath+ f"\\{pdf_type}.svg")
    plt.show()

# -------------------- Plot All Cases --------------------
plot_hist_and_fit(stats_all, legend_labels, pdf_type="all", ylim=(0,0.13))




# -------------------- Plot individual measurements --------------------
def plot_individual_measurements(PDA_case, label):
    plt.figure(figsize=(6, 4))
    handles = []
    labels = []
    for i, entry in enumerate(PDA_case, start=1):
        d = entry["d"]
        # Skip empty entries
        if len(d) == 0:
            continue

        # Histogram
        N, _ = np.histogram(np.log(d), bins=log_d_edges)
        dx = np.diff(np.exp(log_d_edges))
        pdf = N / np.sum(N*dx)


        # Fit
  
        x_fit, pdf_fit, mean_d, std_d = log_normal_fit(d, d_edges)
        h, = plt.plot(x_fit, pdf_fit, '--',
                 label=f'{i}') #label=rf'{i}, $\mu$ {mean_d:.1f} μm, std={std_d:.1f} μm'
        handles.append(h)
        labels.append(f'{i}: {mean_d:.1f}, {std_d:.1f}')
    custom_handle = Line2D([0], [0], color='none')  # invisible
    handles = [custom_handle] + handles
    labels = [rf'no., mean: ($\mu$m), std:($\mu$m)'] + labels
    plt.xscale("log")
    plt.xlabel(r"$d$ [$\mu$m]")
    plt.ylabel(r"p.d.f. [$\mu$m$^{-1}$]")
    plt.ylim(0, 0.15)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title(label)
    plt.legend(handles, labels, loc='upper right', frameon=True,ncols=2,fontsize=8)
    plt.tight_layout()
    plt.savefig(savepath+ f"\\indvidual_{label}.svg")
    plt.show()





# Plot for 0.2%wt
wt_index = casenames.index("050B_0pt2wt")
#plot_individual_measurements(PDA_all[wt_index], "0.2%wt individual measurements")

# Plot for water
water_index = casenames.index("050B_water")
plot_individual_measurements(PDA_all[water_index], "Water individual measurements")
