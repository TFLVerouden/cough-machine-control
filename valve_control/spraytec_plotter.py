import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

import scienceplots


def set_size(width, fraction=1, subplots=(1, 1), aspect=None):
    """Set figure dimensions to avoid scaling in LaTeX.

    From https://jwalton.info/Embed-Publication-Matplotlib-Latex/

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5 ** .5 - 1) / 2

    if aspect is None:
        aspect = golden_ratio

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * aspect * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

# Set the style to "science"
plt.style.use("science")

matplotlib.use("TkAgg")  # Or "Agg", "Qt5Agg", "QtAgg"
path = r"/Users/tommieverouden/Library/CloudStorage/OneDrive-UniversityofTwente/Coughers Team/Detlef_Presentation_4-4-2025/SprayTec_Distributions/1-32_PEO2M_5_numd_av.txt"

data = pd.read_csv(path, delimiter=",", skiprows=66, nrows=60,encoding="latin1",names=["PDF", "Cum", "Diameter"])
print(data["Diameter"])
bin_edges= (data["Diameter"][1:].values + data["Diameter"][:-1].values)/2
bin_edges = np.append(bin_edges, (data["Diameter"].values[-1]-bin_edges[-1])*2)  # Append the last bin edge
print(bin_edges)


bin_width = np.diff(bin_edges)
bin_width = np.append(bin_width, (data["Diameter"].values[-1]-bin_edges[-1])*2)  # Append the last bin width

print(np.shape(data["PDF"][1:]))



area = np.sum(data["Diameter"] * data["PDF"])  # Total area under the curve (sum of bin areas)
data["PDF"] /= area  # Normalize the PDF to make it a probability density function

fig, ax = plt.subplots(figsize=set_size(200))
plt.bar(bin_edges, data["PDF"], label="PDF", color="blue",edgecolor="black", alpha=0.7, width=bin_width,align="edge")
plt.grid()
plt.xscale('log')
plt.yscale('log')
plt.xlim(5,1000)
plt.ylim(1e-6, 1e-2)
plt.xlabel("Diameter (µm)")
plt.ylabel("Probability density (1/µm)")
# plt.show()


fig.savefig('spraytec_example.pdf', format='pdf', bbox_inches='tight')

