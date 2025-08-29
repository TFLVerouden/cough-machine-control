import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
filename = r"D:\Experiments\Processed_Data\RemotePC\\Processed_arrays\\0dot25percentPEO1ml_4th.pkl"
with open(filename, 'rb') as f:
    loaded = pickle.load(f)   

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





plt.figure()
plt.scatter(time,all_avg,c='b', s=1)
plt.scatter(time,all_avg+all_std,c='r',marker='_',s=1)
plt.scatter(time,all_avg-all_std,c='r',marker='_',s=1)
plt.show()


plt.figure()
plt.scatter(all_length,all_avg,c='b', s=1)
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