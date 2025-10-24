import numpy as np 
import matplotlib.pyplot as plt

x= np.arange(250)/1000
y_step= np.concatenate((np.zeros(20),np.ones(80),np.zeros(150))) *100
y_prop = np.concatenate((np.zeros(20),np.linspace(0,1,20),np.ones(50),np.linspace(1,0.8,4),np.ones(10)*0.8, np.linspace(0.8,0,16),np.zeros(130)))*100
plt.figure(figsize=(5,3))
plt.plot(x,y_step,c="k",label="Our valve")
plt.plot(x,y_prop,c="r", label= "Proportional valve")
plt.xlabel("Time (s)")
plt.ylabel("Valve open (%)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(rf"C:\Users\sikke\Documents\universiteit\Master\Thesis\presentation\proportionalvalve.svg")

plt.show()