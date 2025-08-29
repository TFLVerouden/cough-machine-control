import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import re

filename = r"D:\Experiments\Processed_Data\RemotePC\\Processed_arrays\0dot25percentPEO1ml_4th.pkl"
with open(filename, 'rb') as f:
    loaded = pickle.load(f)[0]   

match = re.search(r'\\([^\\]+?)\.pkl$', filename)

if match:
    save_filename = match.group(1)
    print(save_filename)
else:
    print("No match found.")
    
save_path = r"D:\Experiments\Processed_Data\RemotePC\\mean_images"  + save_filename + ".png" 
save_path_filter = r"D:\Experiments\Processed_Data\RemotePC\\mean_images\\"  + save_filename + "_filtered.png" 

scale =15.67
fps =20000
length = len(loaded)
time = np.arange(0,length,1)
n_bins = 50
timesteps = length

all_avg = []
all_std = []
all_median = []
all_length = []
all_timesteps = []
for i in range(timesteps):
    print(i)
    time_data = loaded[i]

    
    avg_list = []
    std_list = []
    median_list = []
    length_list = []
    time_list = []
    if len(time_data)>0:

        for filament in time_data:
            
            
            
            # if len(filament) ==0:
            #     continue
            # filament = filament[0]
            filament = filament[~np.isnan(filament)]
            
            if len(filament) == 0:
                continue  # Skip empty filaments
            print(i)
            time_list.append(i)
            avg_list.append(np.average(filament))
            std_list.append(np.std(filament))
            median_list.append(np.median(filament))
            length_list.append(len(filament))
    all_timesteps.append(time_list)
    all_avg.append(avg_list)
    all_std.append(std_list)
    all_median.append(median_list)
    all_length.append(length_list)
        

avg_flatten = []
median_flatten = []
std_flatten = []
length_flatten = []
time_flatten = []

for avgi,mediani,stdi,leni,timei in zip(all_avg,all_median,all_std,all_length,all_timesteps):
    avg_flatten += avgi
    median_flatten += mediani
    std_flatten += stdi
    length_flatten += leni
    time_flatten += timei


avg_flatten = np.array(avg_flatten)
median_flatten = np.array(median_flatten)
std_flatten = np.array(std_flatten)
length_flatten = np.array(length_flatten)
time_flatten = np.array(time_flatten)

len_mask = length_flatten >100
std_mask = std_flatten <3
median_mask = abs(median_flatten-avg_flatten)<5
mask = (std_mask) & (len_mask) & (median_mask)
plt.figure()
plt.scatter(time_flatten[mask]/fps,avg_flatten[mask]/scale,c='k',s=2)
plt.xlabel('Time (ms)')
plt.ylabel('Mean width (mm)')
plt.tight_layout()
plt.savefig(save_path)
plt.show()

plt.figure()
plt.scatter(length_flatten[mask]/scale,avg_flatten[mask]/scale,c='k',s=1,label= 'Kept')
plt.scatter(length_flatten[~mask]/scale,avg_flatten[~mask]/scale,s=1 ,c='orange',label="Discard")
plt.xlabel('Filament length (mm)')
plt.ylabel('Filament width (mm)')
plt.legend()
plt.tight_layout()
plt.savefig(save_path_filter)
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