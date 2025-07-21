import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
from datetime import datetime

def split_array_by_header_marker(arr, marker='Date-Time'):
    arr = np.array(arr)
    header = arr[:,0]
    rows = arr[:,1:]

    # Find indices where header has the marker
    split_indices = [i for i, val in enumerate(header) if val == marker]
    split_indices.append(len(header))  # include end boundary

    result = []
    for i in range(len(split_indices) - 1):
        start = split_indices[i]
        end = split_indices[i+1]
        section = arr[start:end]
        result.append(section)

    return result

current_dir = os.path.dirname(os.path.abspath(__file__))

path = os.path.join(current_dir,"auto_export_test.txt")
print(path)
save_path = os.path.join(current_dir, "individual_data_files")
file = np.loadtxt(path,dtype=str,delimiter=',')
print(file.shape)
print(file[:,0])

split_sections = split_array_by_header_marker(file)

last_file = split_sections[-1]
time_created= last_file[1,0]
print(time_created)
dt = datetime.strptime(time_created, '%d %b %Y %H:%M:%S.%f')

# Format as YYYY_MM_DD_HH_MM
file_name_time = dt.strftime('%Y_%m_%d_%H_%M')

save_path = os.path.join(save_path,file_name_time + ".txt")
np.savetxt(save_path,last_file,fmt='%s',delimiter=',')



