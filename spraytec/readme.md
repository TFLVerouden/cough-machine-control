The folder averages unweighted has all measurement which are averaged over 1 s as .txt
The folder individual_data_files has all measurements with time information
The folder readme has nice readmes describing the Spraytec
- aerosolconcentration.py plots the aersol concentration (cv) over time for a measurement serie
- conferenceplotter compares different settings with each other
- morgan comparison is written so that we can compare our data with those of Li (2025)
-Spraytec_append_file.txt should not be touched. Here are the data of the spraytec measurements is automatically saved via appending after each measurement. 
- Spraytec_averageplotseries plots the average distribution over a measurement series
- Spraytec_averaging.py plots all individual average distribution over time for a complete measurement series
- Spraytec_fulldataset.py is the first step to save all relevant statistiscs
-Spraytec_log_normalfitting.py allows us to either fit a log normal or gamma distribution to our measurements. It has the functionality to fit both modal and bimodal distributions
-Spraytec_manual_txt_converter.py lets you manually write data from the spraytec_append_file.txt to the folder individual_data_files
-Spraytec_rawdataplotter.py makes the imshows over time with time on x-axis and diameter on y-axis, where the color indicates the number distribution