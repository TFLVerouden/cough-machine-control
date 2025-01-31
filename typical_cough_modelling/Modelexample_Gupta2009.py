import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import Gupta2009 as Gupta


#testing

Tau = np.linspace(0,10,101)

#person A lowest values of Male from Gupta et al 2009

CPFR_A = 3 #L/s
CEV_A = 0.400 #L
PVT_A = 57E-3 #s

cough_A = Gupta.M_model(Tau,PVT_A,CPFR_A,CEV_A)

#person B highest values of Male from Gupta et al 2009

CPFR_B = 8.5 #L/s
CEV_B = 1.6 #L
PVT_B = 96E-3 #s

cough_B = Gupta.M_model(Tau,PVT_B,CPFR_B,CEV_B)

#person C lowest values of feale from Gupta et al 2009

CPFR_C= 1.6 #L/s
CEV_C = 0.25 #L
PVT_C = 57E-3 #s

cough_C = Gupta.M_model(Tau,PVT_C,CPFR_C,CEV_C)

#person D highest values of Male from Gupta et al 2009

CPFR_D = 6 #L/s
CEV_D = 1.25 #L
PVT_D = 110E-3 #s

cough_D = Gupta.M_model(Tau,PVT_D,CPFR_D,CEV_D)

#person E, me based on Gupta et al
PVT_E, CPFR_E, CEV_E = Gupta.estimator("Male",70, 1.89)

cough_E = Gupta.M_model(Tau,PVT_E,CPFR_E,CEV_E)

plt.figure()
plt.plot(Tau,cough_A,label= "Person A, lower bounds male",c='b')
plt.plot(Tau,cough_B,label= "Person B, higher bounds male",c='r')
plt.plot(Tau,cough_C,label= "Person C, lower bounds female",c='g')
plt.plot(Tau,cough_D,label= "Person D, higher bounds female",c='k')
plt.plot(Tau,cough_E,label= "Me",c='orange')
plt.xlabel(r"$\tau$")
plt.ylabel("M")
plt.legend()
plt.show()

