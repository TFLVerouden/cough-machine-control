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

path = os.path.join(current_dir,"SPRAYTEC_APPEND_FILE.txt")

save_base_path = os.path.join(current_dir, "individual_data_files")
file = np.loadtxt(path,dtype=str,delimiter=',')


split_sections = split_array_by_header_marker(file)

only_last_file = input("Do you want so save only the last file? (y/n)").strip().lower()

if only_last_file == "y": 

    last_file = split_sections[-1]
    time_created= last_file[1,0]
    filename= last_file[1,1]
    dt = datetime.strptime(time_created, '%d %b %Y %H:%M:%S.%f')

    # Format as YYYY_MM_DD_HH_MM
    file_name_time = dt.strftime('%Y_%m_%d_%H_%M')

    save_path = os.path.join(save_base_path,file_name_time + "_" + filename + ".txt")
    np.savetxt(save_path,last_file,fmt='%s',delimiter=',')
else: 
    all_files = input("Do you want so save all data (y/n)").strip().lower()
    if all_files =="y":
        for section in split_sections:

            file = section
            time_created= file[1,0]
            filename= file[1,1]
            dt = datetime.strptime(time_created, '%d %b %Y %H:%M:%S.%f')

            # Format as YYYY_MM_DD_HH_MM
            file_name_time = dt.strftime('%Y_%m_%d_%H_%M')

            save_path = os.path.join(save_base_path,file_name_time + "_" + filename + ".txt")
            np.savetxt(save_path,file,fmt='%s',delimiter=',')

    else:
        i=0
        for section in split_sections:
            
            file = section
            time_created= file[1,0]
            filename= file[1,1]

            save_this_file = input(f"This is file {i}, created at {time_created}, do you want to save it? (y/n)").strip().lower()
            if save_this_file== "y":
                dt = datetime.strptime(time_created, '%d %b %Y %H:%M:%S.%f')

                # Format as YYYY_MM_DD_HH_MM
                file_name_time = dt.strftime('%Y_%m_%d_%H_%M')
                
                save_path = os.path.join(save_base_path,file_name_time + "_" + filename + ".txt")
                np.savetxt(save_path,file,fmt='%s',delimiter=',')
            i+=1



