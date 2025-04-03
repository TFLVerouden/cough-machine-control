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

PEO1= pd.read_csv(path1, delimiter=",", skiprows=66, nrows=60,encoding="latin1",names=["PDF", "Cum", "Diameter"])
PEO0dot03= pd.read_csv(path2, delimiter=",", skiprows=66, nrows=60,encoding="latin1",names=["PDF", "Cum", "Diameter"])
PEO0dot25 =pd.read_csv(path3, delimiter=",", skiprows=66, nrows=60,encoding="latin1",names=["PDF", "Cum", "Diameter"])


# print(one_one["Diameter"])
# bin_edges= (one_one["Diameter"][1:].values + one_one["Diameter"][:-1].values)/2
# bin_edges = np.append(bin_edges, (one_one["Diameter"].values[-1]-bin_edges[-1])*2)  # Append the last bin edge
# print(bin_edges)


# bin_width = np.diff(bin_edges)
# bin_width = np.append(bin_width, (one_one["Diameter"].values[-1]-bin_edges[-1])*2)  # Append the last bin width

# print(np.shape(one_one["PDF"][1:]))

# print(one_one.loc[np.argmax(one_one["PDF"]),"Diameter"])

def plotting_func(df,color,label,normalized=True):
    area = np.sum(df["Diameter"] * df["PDF"])  # Total area under the curve (sum of bin areas)
    if normalized:
        df["PDF"] /= area  # Normalize the PDF to make it a probability density function
    plt.step(df["Diameter"],df["PDF"],where= 'mid',color= color,label=label)
    #plt.plot(df["Diameter"],df["PDF"],color= color,label=label)

plt.figure(figsize=(10, 6))
#plotting_func(PEO0dot03,"blue",label ="PEO 2M 0.03%")
#plotting_func(PEO0dot25,"green",label ="PEO 2M 0.25%")
plotting_func(PEO1,"red",label ="PEO 2M 1%")
#plt.bar(bin_edges, one_one["PDF"], label="PDF", color="blue",edgecolor="black", alpha=0.7, width=bin_width,align="edge")
plt.grid()
plt.xscale('log')
plt.xlabel("Diameter (µm)")
plt.ylabel(r"Frequency [%]")
#plt.ylabel("Probability Density (1/µm)")

plt.legend()
plt.savefig('PDF_Spraytec_1percentage.png',dpi=200)
#plt.show()

