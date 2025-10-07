import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes,mark_inset
from matplotlib.ticker import FixedLocator, ScalarFormatter
plt.rcParams.update({
    "font.size": 12  # sets default font size to 14
})
file_a = r"C:\Users\sikke\Documents\GitHub\cough-machine-control\other_side_stuff\modedropletsizes\mode_results_Abe.npz"
file_b = r"C:\Users\sikke\Documents\GitHub\cough-machine-control\other_side_stuff\modedropletsizes\mode_results_Morgan.npz"

# Load the mode data
modes_a = np.load(file_a, allow_pickle=True)
modes_b = np.load(file_b, allow_pickle=True)

# Suppose you already have x-axis arrays
# Example:
x_a = np.array([3, 26, 95, 280,0]) #My deborah numbers: Water 0, 600K 0.2: 3, 0.03: 26, 0.25: 95, 1: 280
x_b = np.array([1.14, 1.8, 2.73,5.98, 0]) #Morgan deborah numbers 44m/s: Water0, 0.05: 1.14, 0.1: 1.8,0.2,:2.73, 0.5: 5.98, 1: 11.4

# Extract mode centers in the same order as x_a/x_b
# We sort the keys to match your x arrays if needed
keys_a = sorted(modes_a.files)
keys_b = sorted(modes_b.files)

y_a = np.array([modes_a[k].item()['mode_center'] for k in keys_a])
yerr_a = np.array([
    [y_a[i] - modes_a[k].item()['mode_left'], modes_a[k].item()['mode_right'] - y_a[i]]
    for i, k in enumerate(keys_a)]
).T  # shape (2, N) for asymmetric errors
y_b = np.array([modes_b[k].item()['mode_center'] for k in keys_b])
yerr_b = np.array([
    [y_b[i] - modes_b[k].item()['mode_left'], modes_b[k].item()['mode_right'] - y_b[i]]
    for i, k in enumerate(keys_b)]
).T


# Plot

fig, ax = plt.subplots(1,2,figsize=(8,6))

# Main plot
ax[1].errorbar(x_a, y_a, yerr=yerr_a, fmt='o', ms=4, capsize=3, label="Sikkema 2025")
ax[1].errorbar(x_b, y_b, yerr=yerr_b, fmt='o', ms=4, capsize=3, label="Li 2025")
#ax.set_xscale('log')   # log for full range
ax[1].set_yscale('log')
ax[1].set_xlabel("De")
ax[0].set_ylabel("Mode diameter (μm)")
ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
#ax.legend()
ax[0].errorbar(x_a, y_a, yerr=yerr_a, fmt='o', ms=4, capsize=3, label="Sikkema 2025")
ax[0].errorbar(x_b, y_b, yerr=yerr_b, fmt='o', ms=4, capsize=3, label="Li 2025")
#ax.set_xscale('log')   # log for full range
ax[0].set_yscale('log')
ax[0].set_xlabel("De")

ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
ax[0].set_xlim(-0.5,30)
ax[0].set_ylim(6,30)

ax[0].text(0.08, 0.92, "a.", transform=ax[0].transAxes, fontsize=12, fontweight="bold", va="bottom", ha="right")
ax[1].text(0.08, 0.92, "b.", transform=ax[1].transAxes, fontsize=12, fontweight="bold", va="bottom", ha="right")
ax[0].yaxis.set_major_locator(FixedLocator([6, 10, 20, 30]))
ax[0].yaxis.set_major_formatter(ScalarFormatter())
# axins.tick_params(axis='y', which='major', labelsize=10)  # optional: adjust tick size
#axins.grid(True, which='both', linestyle='--', linewidth=0.5)


plt.savefig(r"C:\Users\sikke\Documents\GitHub\cough-machine-control\other_side_stuff\modedropletsizes\modedroplet.pdf")
plt.show()