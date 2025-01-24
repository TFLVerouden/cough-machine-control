# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('240327_test1', delimiter=",",skiprows=1)
#0607_2022/050B_0pt5wt_PEO_600k_h1_e10
#0204_2022_surfactant/050B_1wt_PEO_600k_h1_SDS_0pt01wt_out_e1
x = data[:,0]
y = data[:,1]*(600/32000)

ind = y>1


plt.plot(x[ind],y[ind],'b--')
plt.show

Ubulk = max(y)*1e-3/60/2e-4