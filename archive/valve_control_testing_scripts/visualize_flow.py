import propar
import serial
import time
import serial.tools.list_ports
import datetime
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Or try 'Agg', 'Qt5Agg', 'GTK3Agg', etc.

cwd = os.path.abspath(os.path.dirname(__file__))

parent_dir = os.path.dirname(cwd)
print(parent_dir)
# function_dir = os.path.join(parent_dir, 'cough-machine-control')
function_dir = os.path.join(parent_dir, 'functions')
print(function_dir)
sys.path.append(function_dir)
import Gupta2009 as Gupta
import pumpy
from Ximea import Ximea


####Finished loading Modules


filename = r"C:\Users\local2\PycharmProjects\cough-machine-control\valve_control\results\2_5bar50ms.csv"

plotdata = pd.read_csv(filename, delimiter=",", decimal=".",skiprows=9, names=["Time", "Pressure", "Flow"])

plotdata = np.array(plotdata)

print(plotdata)
dt = np.diff(plotdata[:, 0])
mask = plotdata[:, 2] > 0  # finds the first time the flow rate is above 0

t0 = plotdata[mask, 0][0]
peak_ind = np.argmax(plotdata[:, 2])
PVT = plotdata[peak_ind, 0] - t0  # Peak velocity time
CFPR = plotdata[peak_ind, 2]  # Critical flow pressure rate (L/s)
CEV = np.sum(dt * plotdata[1:, 2])  # Cumulative expired volume

t = plotdata[:, 0]
fig, ax1 = plt.subplots()
ax1.plot(t, plotdata[:, 2], 'b-', label="2.5 bar, 50 ms valve opened", marker="o",
         markeredgecolor="k")

ax1.set_xlabel('Time (s)')
ax1.set_xlim(0,0.6)
ax1.legend()
ax1.set_ylabel('Flow rate (L/s)')
#ax1.set_title(f' CFPR: {CFPR:.1f} L/s, PVT: {PVT:.2f} s, CEV: {CEV:.1f} L\n')
ax1.grid()
plt.tight_layout()

plt.savefig(r"C:\Users\local2\PycharmProjects\cough-machine-control\valve_control\results\2_5bar50ms_withouttitle.png",dpi=200)
