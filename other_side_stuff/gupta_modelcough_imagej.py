import matplotlib.pyplot as plt
import numpy as np
path = r"C:\Users\sikke\Documents\universiteit\Master\Thesis\Images\Typical coughs\Resultsmodelgupta.csv"
data = np.loadtxt(path,delimiter=",",skiprows=1)
X,Y = data[:,-2],data[:,-1]

X_0, Y_0 = X[0],Y[0]  #Define origin

X -=X_0
Y -= Y_0
Y = - Y #Change axis
plt.plot(X,Y)
plt.show()