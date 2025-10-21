import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

"""
Loads in a pickle displays the average width over time, and the width vs length of filament
Average all filaments found for one timestep
"""
scale =15.67
fps=20000

filename = r"D:\Experiments\Processed_Data\RemotePC\\Processed_arrays\\0dot25percentPEO1ml_4th.pkl"
with open(filename, 'rb') as f:
    loaded = pickle.load(f)   
loaded = loaded[0]
length = len(loaded)
time = np.arange(0,length,1,dtype=float)
n_bins = 50
timesteps = length

all_avg = np.zeros_like(time)
all_std = np.zeros_like(time)
all_median = np.zeros_like(time)
all_length = np.zeros_like(time)
all_timesteps = np.zeros_like(time)

for i in range(timesteps):
    
    time_data = loaded[i]
    print(len(time_data))
    if len(time_data)>0:

       
        combined = np.concatenate([c.reshape(-1) for c in time_data])
        combined = combined[~np.isnan(combined)]
        if len(combined>0):
            all_avg[i] = np.mean(combined)
            all_std[i] =np.std(combined)
            all_median[i] =np.median(combined)
            all_length[i] = len(combined)
        else:
            all_avg[i] = np.nan
            all_std[i] =np.nan
            all_median[i] =np.nan
            all_length[i] = np.nan

    else:
            all_avg[i] = np.nan
            all_std[i] =np.nan
            all_median[i] =np.nan
            all_length[i] = np.nan


#aspect_ratio>=3:
#area>40:
area= 40/scale/scale
min_width_area =np.sqrt(area/3)
min_width = 1/scale
aspect_y= np.linspace(min_width,1.4,100)
aspect_x = aspect_y *2.8
t= time/20000
plt.figure()
plt.fill_between(t, np.max(np.array([min_width*np.ones(len(all_avg)),(all_avg-all_std)/scale]),axis=0), (all_avg+all_std)/scale, color='C0', alpha=0.3)
plt.scatter(t,all_avg/scale,c='b',s=1)
# plt.scatter(t,all_avg+all_std,c='r',marker='_',s=1)
# plt.scatter(t,all_avg-all_std,c='r',marker='_',s=1)
plt.xlabel("Time (s)")
plt.ylabel("Average filament width (mm)")
plt.hlines(min_width,np.min(t),np.max(t),color="k",linestyles="--")

plt.ylim(0)
plt.savefig(r"C:\Users\sikke\Documents\GitHub\cough-machine-control\filament_width\filamentwidth_0dot25_errorbars.pdf")
#plt.title("Average width over time")
#plt.show()
x = np.linspace(min_width,np.nanmax(all_length/scale), 100)  # avoid division by zero
y = area / x

plt.figure()
plt.scatter(all_length/scale,all_avg/scale,c='b', s=1)
plt.hlines(min_width,3* min_width,np.nanmax(all_length/scale),color="k",linestyles="--")
print(min_width) 
print(np.min(all_avg/scale))

plt.plot(aspect_x,aspect_y,"--",c="g")
plt.xlabel("Length filament (mm)")
plt.ylabel("Width filament (mm)")
plt.savefig(r"C:\Users\sikke\Documents\GitHub\cough-machine-control\filament_width\filamentwidthvslength_0dot25.pdf")

#plt.title("All lengths vs all width")
plt.show()


#     if len(time_data)>0:
#         combined = np.concatenate([c.reshape(-1) for c in time_data])
#         hist_data,bin_edges = np.histogram(combined,bins=n_bins,range=(0,50))
#         hist_data_all[:,i] = hist_data 


# plt.figure(figsize=(9,9))
# plt.imshow(hist_data_all,aspect='auto')
# plt.ylabel('Width (pix)')
# plt.xlabel('Time')
# cbar = plt.colorbar()
# cbar.set_label('Pixel Count')  
# plt.gca().invert_yaxis()
# #plt.show()
# plt.savefig(r"C:\Users\s2557762\Documents\filament_processing\heatmap_images\test.png")
# plt.show()