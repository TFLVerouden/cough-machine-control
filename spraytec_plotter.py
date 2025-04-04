import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.use("TkAgg")  # Or "Agg", "Qt5Agg", "QtAgg"

cwd = os.path.dirname(os.path.abspath(__file__))

path1 = cwd + r"\\experiments\\Spraytec_data_31032025\\1-1_PEO2M_1_numd_av.txt"
path2 = cwd + r"\\experiments\\Spraytec_data_31032025\\1-1_PEO2M_5_numd_av.txt"
path3 = cwd + r"\\experiments\\Spraytec_data_31032025\\1-4_PEO2M_3_numd_av.txt"
path4 = cwd + r"\\experiments\\Spraytec_data_31032025\\1-32_PEO2M_5_numd_av2.txt"
path5= cwd + r"\\experiments\\Spraytec_data_31032025\\1-32_PEO2M_5_numd_av3.txt"

PEO1= pd.read_csv(path1, delimiter=",", skiprows=66, nrows=60,encoding="latin1",names=["PDF", "Cum", "Diameter"])
PEO0dot03= pd.read_csv(path2, delimiter=",", skiprows=66, nrows=60,encoding="latin1",names=["PDF", "Cum", "Diameter"])
PEO0dot25 =pd.read_csv(path3, delimiter=",", skiprows=66, nrows=60,encoding="latin1",names=["PDF", "Cum", "Diameter"])
PEO0dot03_halfway =pd.read_csv(path4, delimiter=",", skiprows=66, nrows=60,encoding="latin1",names=["PDF", "Cum", "Diameter"])
PEO0dot03_end =pd.read_csv(path5, delimiter=",", skiprows=66, nrows=60,encoding="latin1",names=["PDF", "Cum", "Diameter"])

# #This outcommented part probably doesn't work as it shifts everything one place, but don't want to delete it for now.
# bin_edges= (one_one["Diameter"][1:].values + one_one["Diameter"][:-1].values)/2
# bin_edges = np.append(bin_edges, (one_one["Diameter"].values[-1]-bin_edges[-1])*2)  # Append the last bin edge
# bin_width = np.diff(bin_edges)
# bin_width = np.append(bin_width, (one_one["Diameter"].values[-1]-bin_edges[-1])*2)  # Append the last bin width
#plt.bar(bin_edges, one_one["PDF"], label="PDF", color="blue",edgecolor="black", alpha=0.7, width=bin_width,align="edge")


def plotting_func(df,color,label,normalized=True):
    area = np.sum(df["Diameter"] * df["PDF"])  # Total area under the curve (sum of bin areas)
    if normalized:
        df["PDF"] /= area  # Normalize the PDF to make it a probability density function
        plt.ylabel("Probability Density (1/µm)")
    if not normalized:
        df["PDF"] *= 100
    plt.step(df["Diameter"],df["PDF"],where= 'mid',color= color,label=label)
    plt.ylabel(r"Frequency [%]")


plt.figure(figsize=(10, 6))
plotting_func(PEO0dot03,"blue",label =r"PEO 2M 0.03% begin",normalized=False)
plotting_func(PEO0dot03_halfway,"green",label =r"PEO 2M 0.03% intermediate",normalized=False)
plotting_func(PEO0dot03_end,"red",label =r"PEO 2M 0.03% end",normalized=False)


#plotting_func(PEO0dot25,"green",label ="PEO 2M 0.25%",normalized=False)
#plotting_func(PEO1,"red",label ="PEO 2M 1%",normalized=False)
plt.grid()
plt.xscale('log')
plt.xlabel("Diameter (µm)")
plt.ylabel(r"Frequency [%]")


plt.legend()
#plt.savefig('PDF_Spraytec_time_dependence.png',dpi=200)
#plt.show()

