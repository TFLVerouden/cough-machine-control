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
from scipy.special import gamma as gamma_func
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
#we use this fur future compatibility
pd.set_option('future.no_silent_downcasting', True)

import sys
#plt.style.use("tableau-colorblind10")
cwd = os.path.dirname(os.path.abspath(__file__))
print(cwd)
parent_dir = os.path.dirname(cwd)
function_dir = os.path.join(parent_dir,'functions')

sys.path.append(function_dir)
from cvd_check import set_cvd_friendly_colors

colors = set_cvd_friendly_colors()
plt.rcParams.update({'font.size': 14})

#FINDING THE FILES

save_path = os.path.join(cwd,"results_spraytec","Fits")
print(f"Save path {save_path}")


txt_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')]
pattern = re.compile(rf"average_{re.escape(keyphrase)}_\d+(?:_.*)?\.txt")

# Filter matching files
matching_files = [f for f in txt_files if pattern.search(os.path.basename(f))]
save_path = os.path.join(save_path,keyphrase)

# Create folder if it doesn't exist
os.makedirs(save_path, exist_ok=True)

def log_normal(bin_centers, mu, sigma):
    return 1/ (bin_centers *sigma * np.sqrt(2*np.pi)) *np.exp(-0.5* ((np.log(bin_centers)-mu)/sigma)**2)



# def gamma_pdf(bin_centers, m, n):
#     pdf = (bin_centers**(m-1) * np.exp(-bin_centers/n)) / (gamma_func(m) * n**m)
#     return pdf

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


#find peaks
def fitting(n_pdf,bin_centers,mode="ln"):
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



i=0
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
    n_pdf = n_percentages  /sum(bin_widths*n_percentages)
    mean_d = np.sum(bin_centers * n_percentages) / 100
    
    mode_d = bin_centers[np.argmax(n_pdf)]
    min_records =5
    max_records = 200
    weights = num_records
    max_n_check = np.max(n_percentages)
    max_n_limit = 75
    mask = (weights> min_records) & (weights<max_records) & (max_n_check<max_n_limit)

    if mask and i<6:
        fitpointsfactor =3
       
        fit_x= np.logspace(np.min(np.log10(bin_centers)),np.max(np.log10(bin_centers)),len(bin_centers)*fitpointsfactor)
   
        param_ln, pcov = curve_fit(log_normal, bin_centers, n_pdf)
        mu,sigma =param_ln
        fitted_ln=log_normal(fit_x, *param_ln)
        fitted_ln_per = fitted_ln/ sum(fitted_ln)*100 *fitpointsfactor 
        lower_bounds = [0.05, 0, 0, 0, 0]  # w, mu1, sigma1, mu2, sigma2
        upper_bounds = [0.95, np.inf, np.inf, np.inf, np.inf]
        initial_guess = [0.5, 1.0, 0.5, 2.0, 1.0]
        initial_guess_gamma = [0.5, 5.0, 5, 10.0, 10.0]
        param_2ln, pcov = curve_fit(two_log_normals, bin_centers, n_pdf,bounds=(lower_bounds,upper_bounds),p0=initial_guess) 
        w,mu1,sigma1,mu2,sigma2 = param_2ln
        fitted_2ln = two_log_normals(fit_x,*param_2ln)
        fitted_2ln_per = fitted_2ln / sum(fitted_2ln)*100 *fitpointsfactor

        #gammas
        # param_2gamma, pcov = curve_fit(two_gamma, bin_centers, n_pdf,bounds=(lower_bounds,upper_bounds),p0=initial_guess_gamma) 
        # w_gamma,m1,n1,m2,n2 = param_2gamma
        # fitted_2gamma = two_gamma(fit_x,*param_2gamma)
        # fitted_2gamma_per = fitted_2gamma / sum(fitted_2gamma)*100 *fitpointsfactor

        
        #print(pdf_values)
        #plt.scatter(bin_centers, n_percentages,color=colors[i],label= "Distribution")

        # mode= "gamma"
        # x,pdf,per,n_peaks = fitting(n_pdf,bin_centers,mode=mode)
        #plt.plot(x,per,label= f"Gamma fit",linestyle= "-",color=colors[1])
        mode= "ln"
        x,pdf,per,n_peaks = fitting(n_pdf,bin_centers,mode=mode)

        plt.plot(x,per,label= f"Log-normal fit",linestyle= "-",color=colors[i])
        i+=1
        # plt.plot(fit_x,fitted_2ln,label=f"{w:.2f},{mu1:.2f},{sigma1:.2f},{mu2:.2f},{sigma2:.2f}")
        # plt.plot(fit_x,fitted_2gamma,label=f"{w_gamma:.2f},{m1:.2f},{n1:.2f},{m2:.2f},{n2:.2f}")
        # #plt.title(param_2ln)
        

        
        
       
        
    # Add labels
#plt.legend()
plt.xscale('log')

plt.xlabel(r"D ($\mu$m)")
plt.ylabel("Number distribution (%)")


plt.xscale('log')
#plt.yscale('log')

plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.ylim(1e-1)
plt.xlim(bin_edges[0],bin_edges[-1])
print(f"filename: {filename}")
full_save_path = os.path.join(save_path,filename)
print(f"full path: {full_save_path}")
plt.tight_layout()
#plt.legend()
plt.savefig(full_save_path+".svg")
plt.show()
    
