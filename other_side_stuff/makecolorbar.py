import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.cm import viridis, ScalarMappable

# Compute cap
cap = 50 / 15.67  # ≈ 3.189

# Create a custom colormap where 0 is white
viridis_colors = viridis(np.linspace(0, 1, 256))
viridis_colors[0] = [1, 1, 1, 1]  # make lowest value white
custom_cmap = ListedColormap(viridis_colors)

# Normalize range
norm = Normalize(vmin=0, vmax=cap)

# Scalar mappable for colorbar
sm = ScalarMappable(cmap=custom_cmap, norm=norm)
sm.set_array([])

# Create a figure and a dedicated Axes for colorbar
fig, ax = plt.subplots(figsize=(1, 6))
fig.subplots_adjust(right=0.5)

# Add colorbar in the given Axes
cbar = fig.colorbar(sm, cax=ax, orientation='vertical',extend="max")
cbar.set_label(f'Width (mm)')

plt.show()
