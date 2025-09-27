import numpy as np
import matplotlib.pyplot as plt

def generate_semicircle(center_x, center_y, radius, stepsize=0.1):
    """
    generates coordinates for a semicircle, centered at center_x, center_y
    """        

    x = np.arange(center_x, center_x+radius+stepsize, stepsize)
    y = np.sqrt(radius**2 - x**2)

    # since each x value has two corresponding y-values, duplicate x-axis.
    # [::-1] is required to have the correct order of elements for plt.plot. 
    x = np.concatenate([x,x[::-1]])

    # concatenate y and flipped y. 
    y = np.concatenate([y,-y[::-1]])

    return -x, -y + center_y
a =10
b= 75
x,y = generate_semicircle(0,0,10)
plt.rcParams["font.size"] =16
plt.figure()
plt.gca().set_aspect('equal', adjustable='box')
plt.figure()
plt.hlines(10,0,75)
plt.hlines(-10,0,75)
plt.vlines(75,-10,10)
# plt.xlim(-10,75)
# plt.ylim(-10,10)
plt.xlabel("Vx (m/s)")
plt.ylabel("Vy (m/s)")
plt.plot(x,y)
plt.grid(linestyle="--")
plt.tight_layout()
plt.show()