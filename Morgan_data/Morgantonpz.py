import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from scipy import stats
from scipy.optimize import curve_fit
import sys
from scipy.special import gamma as gamma_func
from scipy.stats import lognorm
from scipy.signal import find_peaks

cwd = os.path.dirname(os.path.abspath(__file__))
print(cwd)
parent_dir = os.path.dirname(cwd)
function_dir = os.path.join(parent_dir,'functions')

sys.path.append(function_dir)
from cvd_check import set_cvd_friendly_colors

colors = set_cvd_friendly_colors()

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
    print(casename)
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
    #Spraytec bins
    ### Get same bins as us
    file = "spraytec/results_spraytec/Serie_Averages/npz_files/water_1ml_1dot5bar_80ms.npz"
    data = np.load(file,allow_pickle=True)
    bins = data['bins']
    bin_widths= data['bin_widths']
    d_edges= np.append(bins,bins[-1]+bin_widths[-1])
    d_bins =(d_edges[:-1] + d_edges[1:]) / 2
    # Histogram bins
    """
    #This is how morgan defined here bines
    log_d_edges = np.arange(0, 4.8 + 0.3, 0.3)  # log-space edges of Morgan
    
    d_edges = np.exp(log_d_edges) #converted to linear space
    # Compute PDFs
    dx = np.diff(d_edges) #Bin_widths in linear space
    d_bins = (d_edges[:-1] + d_edges[1:]) / 2 #bin_centers
    """
    pdf_all,edges = np.histogram(d_all, bins=d_edges)#[0] #histogram based on the edges of Morgan

    dx = bin_widths
    pdf_all = pdf_all / np.sum(pdf_all*dx) #normalizing like a PDF
    

    stats_all.append({
        "d_all": d_all,
        "pdf_all": pdf_all,
    })
    
   
    PDA_all.append(PDA_case)

def log_normal(bin_centers, mu, sigma):
    return 1/ (bin_centers *sigma * np.sqrt(2*np.pi)) *np.exp(-0.5* ((np.log(bin_centers)-mu)/sigma)**2)

def gamma_pdf(bin_centers, m, n, mu=0):
    """3-parameter gamma PDF: shape m, scale n, location mu"""
    x = bin_centers - mu
    pdf = np.zeros_like(bin_centers, dtype=float)
    mask = x > 0
    pdf[mask] = (x[mask]**(m-1) * np.exp(-x[mask]/n)) / (gamma_func(m) * n**m)
    return pdf
def two_gamma(bin_centers, w, n1, m1, n2,m2):
    """
    2-component gamma mixture for Villermaux distributions.

    Parameters:
    - x: data points
    - w: weight of first component (0 < w < 1)
    - n1: scale of first gamma
    - n2: scale of second gamma
    - fixed_m: if not None, fix m1 = m2 = fixed_m
 
    """




    return w * gamma_pdf(bin_centers, m1, n1) + (1 - w) * gamma_pdf(bin_centers, m2, n2)

def two_log_normals(x, w, mu1,sigma1,mu2,sigma2):
    """
    2-component lognormal mixture with fixed shape parameter m.
    Parameters:
    - x: data points
    - w: weight of first component
    - n1: scale of first gamma
    - n2: scale of second gamma
    - m_fixed: fixed shape parameter (default 4)
    """
    
    return w * log_normal(x, mu1, sigma1) + (1 - w) * log_normal(x, mu2, sigma2)   
def fitting(n_pdf,bin_centers,mode="ln"):
    n_percentages = n_pdf / sum(n_pdf)*100 
    fitpointsfactor =3
    mean_d = np.sum(bin_centers * n_percentages) / 100

    mode_d = bin_centers[np.argmax(n_pdf)]
    fit_x= np.logspace(np.min(np.log10(bin_centers)),np.max(np.log10(bin_centers)),len(bin_centers)*fitpointsfactor)
    peaks, stats = find_peaks(n_pdf, distance=3, height=0, width=0)
    ratio_prom = max(stats['prominences'])/ min(stats['prominences'])
    n_peaks =len(peaks)
    if n_peaks>1:
        if ratio_prom >5:
            n_peaks =1
        else: 
            n_peaks =2
    if mode == "ln":
        if n_peaks ==1: #one lognormal fit
            param_ln, pcov = curve_fit(log_normal, bin_centers, n_pdf)
            mu,sigma =param_ln
            fit_pdf=log_normal(fit_x, *param_ln)
            fit_per = fit_pdf/ sum(fit_pdf)*100 *fitpointsfactor 
        elif n_peaks==2: #two lognormals
            lower_bounds = [0.01, 0, 0, 0, 0]  # w, mu1, sigma1, mu2, sigma2
            upper_bounds = [0.99, np.inf, np.inf, np.inf, np.inf]
            initial_guess = [0.5, np.log(mode_d), 0.5,np.log(mean_d), 1.0]
            
            param_2ln, pcov = curve_fit(two_log_normals, bin_centers, n_pdf,bounds=(lower_bounds,upper_bounds),p0=initial_guess) 
            fit_pdf = two_log_normals(fit_x,*param_2ln)
            fit_per = fit_pdf / sum(fit_pdf)*100 *fitpointsfactor
    else:
        if n_peaks==1:
            param_gamma, pcov = curve_fit(gamma_pdf, bin_centers, n_pdf)
            
            fit_pdf=gamma_pdf(fit_x, *param_gamma)
            fit_per = fit_pdf/ sum(fit_pdf)*100 *fitpointsfactor 
        elif n_peaks==2:
            initial_guess_gamma = [0.5, 5.0, 5, 10.0, 10.0]
            lower_bounds = [0.01, 0, 0, 0, 0]  # w, mu1, sigma1, mu2, sigma2
            upper_bounds = [0.99, np.inf, np.inf, np.inf, np.inf]
            param_2gamma, pcov = curve_fit(two_gamma, bin_centers, n_pdf,bounds=(lower_bounds,upper_bounds),p0=initial_guess_gamma) 
            w_gamma,m1,n1,m2,n2 = param_2gamma
            fit_pdf = two_gamma(fit_x,*param_2gamma)
            fit_per = fit_pdf / sum(fit_pdf)*100 *fitpointsfactor

    return fit_x,fit_pdf,fit_per,n_peaks



# -------------------- Plot Function --------------------
def plot_hist_and_fit(stats_all, label_list, pdf_type="all", ylim=(0, 0.13)):
    plt.figure(figsize=(6,4))
    for cc, stat in enumerate(stats_all):
        pdf = stat[f'pdf_{pdf_type}']

        
        percentage = pdf/ sum(pdf)*100
        plt.step(d_bins, percentage, where='mid',color=colors[cc],label=legend_labels[cc]) # plotting the histogram as bin_centers in the middle


        #plt.plot(d_bins, villermaux_mixture_fixed(d_bins, *popt_g), '-x', color=colors[cc],
                #label=f'Mixture fit:  m1={m_fixed:.2f} m2={m_fixed:.2f}, lsq: {residuals_gamma:.5f}')

        # plt.plot(x_fit, pdf_fit, ':',color = colors[cc],
        #          label=f'Calculated {label_list[cc]}, \n mu={mu_d:.1f} μm, \n std={sigma_d:.1f} μm') #plotting the fit
        cwd = os.path.dirname(os.path.abspath(__file__))
        parent = os.path.dirname(cwd)
        #C:\Users\sikke\Documents\GitHub\cough-machine-control\spraytec\results_spraytec\Serie_Averages\npz_files
        save_path = os.path.join(parent,"spraytec","results_spraytec","Serie_Averages")
        series_savepath = os.path.join(save_path,"npz_files")
        print(series_savepath)
        full_series_savepath = os.path.join(series_savepath,legend_labels[cc])
        np.savez(full_series_savepath,n_percentages=percentage,bins=d_edges[:-1],bin_widths=dx)
    plt.title(f"{pdf_type}")
    plt.xscale("log")
    plt.xlabel(r"$d$ [$\mu$m]")
    plt.ylabel(r"p.d.f. [$\mu$m$^{-1}$]")
    #plt.ylim(*ylim)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=14, frameon=False)
    plt.tight_layout()
    #plt.savefig(savepath+ f"\\{pdf_type}.svg")
    plt.show()

# -------------------- Plot All Cases --------------------
#plot_hist_and_fit(stats_all, legend_labels, pdf_type="all", ylim=(0,0.13))




# -------------------- Plot individual measurements --------------------



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
        file = "spraytec/results_spraytec/Serie_Averages/npz_files/water_1ml_1dot5bar_80ms.npz"
        data = np.load(file,allow_pickle=True)
        bins = data['bins']
        bin_widths= data['bin_widths']
        d_edges= np.append(bins,bins[-1]+bin_widths[-1])
        d_bins =(d_edges[:-1] + d_edges[1:]) / 2
        # Histogram bins
        """
        #This is how morgan defined here bines
        log_d_edges = np.arange(0, 4.8 + 0.3, 0.3)  # log-space edges of Morgan
        
        d_edges = np.exp(log_d_edges) #converted to linear space
        # Compute PDFs
        dx = np.diff(d_edges) #Bin_widths in linear space
        d_bins = (d_edges[:-1] + d_edges[1:]) / 2 #bin_centers
        """
        pdf_all,edges = np.histogram(d, bins=d_edges)#[0] #histogram based on the edges of Morgan

        dx = bin_widths
        n_pdf = pdf_all / np.sum(pdf_all*dx) #normalizing like a PDF
    

        # Fit
  
        fit_x,fit_pdf,fit_per,n_peaks = fitting(n_pdf,d_bins,mode="ln")
        h, = plt.plot(fit_x, fit_per, '--',
                 label=f'{i}') #label=rf'{i}, $\mu$ {mean_d:.1f} μm, std={std_d:.1f} μm'

    plt.xscale("log")
    plt.xlabel(r"$d$ ($\mu$m)")
    plt.ylabel(r"Number distribution ($\%$)")
    #plt.ylim(0, 0.15)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    #plt.title(label)
    #plt.legend(handles, labels, loc='upper right', frameon=True,ncols=2,fontsize=8)
    plt.tight_layout()
    print(savepath)
    plt.savefig(savepath+ f"\\indvidual_{label}.svg")
    plt.show()





# Plot for 0.2%wt
wt_index = casenames.index("050B_0pt2wt")
plot_individual_measurements(PDA_all[wt_index], "0.2%wt individual measurements")

# Plot for water
water_index = casenames.index("050B_water")
plot_individual_measurements(PDA_all[water_index], "Water individual measurements")




