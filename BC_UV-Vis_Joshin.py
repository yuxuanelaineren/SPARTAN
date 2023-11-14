# SPARTAN UV-VIS DATA ANALYSIS CODE
# Code Author: Joshin Kumar

import os
import pandas as pd
import numpy as np

# set the directory path where the Excel files are located
Metadata_directory_path = r'C:/Users/renyuxuan/Desktop/Research/Black Carbon/Analysis-10May2023_Joshin\ALL Metadata'

# initialize an empty list to store the dataframes
dataframes = []

# loop through all the files in the directory with .xlsx extension
for file_name in os.listdir(Metadata_directory_path):
    if file_name.endswith('.xlsx'):
        file_path = os.path.join(Metadata_directory_path, file_name)
        df = pd.read_excel(file_path)
        dataframes.append(df)

# combine all the dataframes into a single dataframe
combined_df = pd.concat(dataframes, ignore_index=True)

# # drop irrelevant columns
combined_df.drop(labels=["Shipment ID (Date)", "Cartridge Number", "Barcode", "Project ID", "Lot ID", "Comments"], axis=1, inplace=True)

# Add PM2.5 column
combined_df["PM2.5(ug/m3)"] = combined_df["Mass collected on filter (ug)"]/combined_df["Sampled volume (m3)"]

# Select PM2.5 filters only and drop NaN rows in PM2.5 dataframe
combined_df_PM25 = combined_df[combined_df["Filter Type"]=="PM2.5"]
# Replace empty string and negative values with NaN
combined_df_PM25.replace('', np.nan, inplace=True)
combined_df_PM25[combined_df_PM25["PM2.5(ug/m3)"]<0] = np.nan
# Drop NaN, 0, negative and infinity values
combined_df_PM25 = combined_df_PM25.replace([np.nan, np.inf, -np.inf, 0], np.nan).dropna()


# Add Blank averaged file row to the PM2.5 dataframe
# ADD the new BLANK row as a dictionary
new_row = {'Filter ID': 'AAAA-BLNK-X', 'Analysis ID': 'AAAA-BLNK-X', 'Filter Type': 'NF', 'Sampling Start Date': '0', 'Sampling End Date': '0', 'Mass collected on filter (ug)': '1', 'Sampled volume (m3)': '1', 'PM2.5(ug/m3)': '1'}
# append the new row to the bottom of the dataframe
combined_df_PM25 = combined_df_PM25.append(new_row, ignore_index=True)
display(combined_df_PM25)

# Select Blank filters only
combined_df_Blank = combined_df[combined_df["Filter Type"]=="FB"]
display(combined_df_Blank)

#### CHECKING ALL REFLECTANCE FILES AVAILABLE IN PM2.5 DATA AND UV-VIS
import os
import pandas as pd

# Set directory path and load dataframe
Reflectance_directory_path = r"C:\Users\joshi\Desktop\SPARTAN Meeting 2023\Analysis-10May2023\ALL Reflectance"

# Create a directory to move files without PM2.5 data
new_dir = os.path.join(Reflectance_directory_path, "Files without PM2.5 data")
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

# Loop through each file in directory
for filename in os.listdir(Reflectance_directory_path):
    # Check if file is an excel file
    if filename.endswith(".csv"):
        # Get initial part of filename
        filter_id = filename.split(".")[0]#.split("-")[0]

        # Check if filter ID is in combined_df_PM25
        if filter_id not in combined_df_PM25["Filter ID"].values:
            # Move file to new directory
            src = os.path.join(Reflectance_directory_path, filename)
            dst = os.path.join(new_dir, filename)
            os.rename(src, dst)

#### CHECKING ALL TRANSMITTANCE FILES AVAILABLE IN PM2.5 DATA AND UV-VIS

import os
import pandas as pd

# Set directory path and load dataframe
Transmittance_directory_path = r"C:\Users\joshi\Desktop\SPARTAN Meeting 2023\Analysis-10May2023\ALL Transmittance"

# Create a directory to move files without PM2.5 data
new_dir = os.path.join(Transmittance_directory_path, "Files without PM2.5 data")
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

# Loop through each file in directory
for filename in os.listdir(Transmittance_directory_path):
    # Check if file is an excel file
    if filename.endswith(".csv"):
        # Get initial part of filename
        filter_id = filename.split(".")[0]#.split("-")[0]

        # Check if filter ID is in combined_df_PM25
        if filter_id not in combined_df_PM25["Filter ID"].values:
            # Move file to new directory
            src = os.path.join(Transmittance_directory_path, filename)
            dst = os.path.join(new_dir, filename)
            os.rename(src, dst)

# Save the List of all the pysical R and T filters datafiles which do not have PM2.5 filter data row
# (Ask Chris/Randall for it!) and include them in the analysis
import os

output_file = "List_of_Filters_with_Missing_Metadata.txt"

file_list = os.listdir(new_dir)

with open(output_file, "w") as file:
    for file_name in file_list:
        file.write(file_name + "\n")

# List of Files present in directory1 but NOT in directory2
# Know the filenames and MANUALLY move them from the directory

import os

# Undo the comments and swap the directory names and re-run the cell to complete the check
directory1 = Transmittance_directory_path #Reflectance_directory_path
directory2 = Reflectance_directory_path #Transmittance_directory_path

# Get the list of filenames in Directory1
files1 = os.listdir(directory1)

# Get the list of filenames in Directory2
files2 = os.listdir(directory2)

# Filter the list of files in Directory1 to only include .csv files
files1 = [f for f in files1 if f.endswith('.csv')]

# Filter the list of files in Directory2 to only include .csv files
files2 = [f for f in files2 if f.endswith('.csv')]

# Find the list of files in Directory1 but not in Directory2
files_not_in_directory2 = list(set(files1) - set(files2))

print(files_not_in_directory2)

import os

# Undo the comments and swap the directory names and re-run the cell to complete the check
directory1 = Reflectance_directory_path#Reflectance_directory_path
directory2 = Transmittance_directory_path#Transmittance_directory_path

# Get the list of filenames in Directory1
files1 = os.listdir(directory1)

# Get the list of filenames in Directory2
files2 = os.listdir(directory2)

# Filter the list of files in Directory1 to only include .csv files
files1 = [f for f in files1 if f.endswith('.csv')]

# Filter the list of files in Directory2 to only include .csv files
files2 = [f for f in files2 if f.endswith('.csv')]

# Find the list of files in Directory1 but not in Directory2
files_not_in_directory2 = list(set(files1) - set(files2))

print(files_not_in_directory2)

###### Average together all the Blank files and just keep 1 Averaged blank file in Transmittance and Reflectance directory

import os
import pandas as pd
import os
import glob
import pandas as pd

# Define the directory path where BLANK files are stored
directory_path = os.path.join(Reflectance_directory_path, "Files without PM2.5 data") #"/path/to/directory"

# Initialize the empty list to hold dataframes
df_list = []

# Loop through all csv files in the directory
for file in os.listdir(directory_path):
    if file.startswith("BLANK") and file.endswith(".csv"):
        # Read the csv file into a dataframe
        filepath = os.path.join(directory_path, file)
        df = pd.read_csv(filepath, sep=",")
        # Append the dataframe to the list
        df_list.append(df)

# Combine all dataframes into a single dataframe
combined_df = pd.concat(df_list, axis=0)

# Group by the "nm" column and calculate the mean of the "%R" column
averaged_df = combined_df.groupby("nm").mean().reset_index()

# Save the averaged dataframe to a csv file
# averaged_df.drop(averaged_df.columns[2], axis=1, inplace=True)
averaged_df = averaged_df.sort_values(by="nm", ascending=False)

averaged_df.to_csv(os.path.join(Reflectance_directory_path, "AAAA-BLNK-X.Sample.Raw.csv"), index=False)
display("Reflectance Blank Averaged", averaged_df)
#############
# Define the directory path where BLANK files are stored
directory_path = os.path.join(Transmittance_directory_path, "Files without PM2.5 data") #"/path/to/directory"

# Initialize the empty list to hold dataframes
df_list = []

# Loop through all csv files in the directory
for file in os.listdir(directory_path):
    if file.startswith("BLANK") and file.endswith(".csv"):
        # Read the csv file into a dataframe
        filepath = os.path.join(directory_path, file)
        df = pd.read_csv(filepath, sep=",")
        # Append the dataframe to the list
        df_list.append(df)

# Combine all dataframes into a single dataframe
combined_df = pd.concat(df_list, axis=0)

# Group by the "nm" column and calculate the mean of the "%R" column
averaged_df = combined_df.groupby("nm").mean().reset_index()

# Save the averaged dataframe to a csv file
# averaged_df.drop(averaged_df.columns[2], axis=1, inplace=True)
averaged_df = averaged_df.sort_values(by="nm", ascending=False)

averaged_df.to_csv(os.path.join(Transmittance_directory_path, "AAAA-BLNK-X.Sample.Raw.csv"), index=False)
display("Transmittance Blank Averaged", averaged_df)

#### Remove the irrelevant rows in the PM2.5 dataframe which does not have a physical T and R files ####

import os

directory_path = Reflectance_directory_path#"/path/to/directory"

# Get a list of filenames present in the directory
dir_filenames = os.listdir(directory_path)

# Extract the filter ID from the filename using string manipulation techniques
dir_filter_ids = [filename.split(".")[0] for filename in dir_filenames]

# Create a boolean mask for the rows in the dataframe based on whether the filter ID is present in the list of filenames
mask = combined_df_PM25["Filter ID"].isin(dir_filter_ids)

# Use the boolean mask to filter out the rows from the dataframe
combined_df_PM25_filtered = combined_df_PM25[mask]

# Sort the combined_df_PM25_filtered dataframe on Filter ID
combined_df_PM25_filtered = combined_df_PM25_filtered.sort_values(by=['Filter ID'], ascending=True)  # sort by Filter ID column

combined_df_PM25_filtered

############################# UV-VIS DATA ANALYSIS RUN CODE

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 09:50:37 2021

@author: Nishit and Joshin

Update1:7/16/2021 by Joshin | Few additional tweaks
Update2:7/16/2021 By Joshin | Code to make a dashboard for data visulization
Update3:3/20/2022 By Joshin | Added Length Consistency Error Debugger
"""

#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

##FilterChangeReq
Path = str(r'C:\Users/renyuxuan/Desktop/Research/Black Carbon/Analysis-10May2023_Joshin\ALL Transmittance')
blank_data_files = [f for f in os.listdir(Path) if 'Blank' in f]
d = 25*1e-3

def load_files_t(filenames):
 	for filename in filenames:
         yield pd.read_csv(Path +"\\"+ filename, usecols=[' %T'])

T_data_files = [f for f in os.listdir(Path) if 'Sample.Raw' in f]
#T_data_files = sorted(T_data_files)
T_Data = pd.concat(load_files_t(T_data_files), axis=1)
T = T_Data.to_numpy()

blankt_data_files = [f for f in os.listdir(Path) if 'BLNK' in f]
Blank_t_Data = pd.concat(load_files_t(blankt_data_files), axis=1)
Blank_t_average = Blank_t_Data.mean(axis=1)
Blank_t = Blank_t_average.to_numpy()

Ref_t_value = [T[x,:]/Blank_t[x] for x in range(len(Blank_t))] #Normalizing Transmittance
Ref_t_value = np.asarray(Ref_t_value)

Ref_t_value[Ref_t_value > 1] = 1 # Reducing all unrealistic values above 1 to 1.
Ref_t_value[Ref_t_value<0] = 0.0000001

# Blank_t_Data = pd.concat(load_files_t(blank_data_files), axis=1)
# Blank_t_average = Blank_t_Data.mean(axis=1)

# IDBD_t_data_files = [f for f in os.listdir(Path) if 'IDBD' in f] #Selecting particular data files based on location
# IDBD_t_Data = pd.concat(load_files_t(IDBD_t_data_files), axis=1)

# ILNZ_t_data_files = [f for f in os.listdir(Path) if 'ILNZ' in f]
# ILNZ_t_Data = pd.concat(load_files_t(ILNZ_t_data_files), axis=1)

# ZAPR_t_data_files = [f for f in os.listdir(Path) if 'ZAPR' in f]
# ZAPR_t_Data = pd.concat(load_files_t(ZAPR_t_data_files), axis=1)

##FilterChangeReq
Path2 = str(r'C:\Users/renyuxuan/Desktop/Research/Black Carbon/Analysis-10May2023_Joshin\ALL Reflectance')

def load_files_r(filenames):
	for filename in filenames:
		yield pd.read_csv(Path2 +"\\"+ filename, usecols=[' %R'])


R_data_files = [f for f in os.listdir(Path2) if 'Sample.Raw' in f]
#R_data_files = sorted(R_data_files)
R_Data = pd.concat(load_files_r(R_data_files), axis=1)
R = R_Data.to_numpy()

blank_data_files = [f for f in os.listdir(Path2) if 'BLNK' in f]
Blank_r_Data = pd.concat(load_files_r(blank_data_files), axis=1)
Blank_r_average = Blank_r_Data.mean(axis=1)
Blank_r = Blank_r_average.to_numpy()

Ref_r_value = [R[x,:]/Blank_r[x] for x in range(len(Blank_r))] #Normalizing Reflectance
Ref_r_value = np.asarray(Ref_r_value)

Ref_r_value[Ref_r_value >= 1] = 0.99999999 # to avoid zero or negative value inside log

ODs = np.log((1-Ref_r_value)/Ref_t_value)
ODs = np.abs(ODs)

##FilterChangeReq
Path3 = str(r'C:\Users/renyuxuan/Desktop/Research/Black Carbon/Analysis-10May2023_Joshin\Results')
Mass = pd.read_excel(Path3 +"\\"+ 'ALL PM2.5.xlsx') # Mass data has been provided by Brenna. Please check her email for relevant files
Mass_r = Mass.to_numpy()

######## Update3: 03/20/2022 By Joshin | Length Consistency Error Debugger b/w ALL R, ALL T and ALL PM2.5
smaller_length = len(R_data_files) # Enter the smaller of the lengths if Code does not run due to unequal lenghts
for i in range(smaller_length):
    if (Mass["Filter ID"][i][0:11] != R_data_files[i][0:11]):
        print("Length Consistency Check Error: Check around this PM2.5 File and Reflectance File in the folder")
        print(i+1)
        print(Mass["Filter ID"][i][0:11])
        print(R_data_files[i][0:11])
        import sys
        sys.exit("Code Stopped Abruptly!")
########



# ODs_f = pd.read_excel(Path3 + 'ODs.xlsx')
# ODs_f = ODs_f.to_numpy()

# ODs_r = np.delete(ODs_f, [6,14,22,30], axis=1) # Deleting bad data files which had negative mass values (erroroneous measurements from Dr. Martin's group)
# Mass_r = np.delete(Mass_f, [6,14,22,30], axis=0)

# Code to get Lambda column names 300 to 900nm Range
##FilterChangeReq
Lambda_values = pd.read_csv(Path2 +"\\"+ 'AAAA-BLNK-X.Sample.Raw.csv', usecols=['nm'])
Lambda_values = Lambda_values.values.flatten()

MAC_r = [((0.48*(ODs[:,x])**1.32)/Mass_r[x,5])*(np.pi*(d**2)/4)*1e6 for x in range(len(Mass_r))]
MAC_r = np.asarray(MAC_r)
MAC_r = pd.DataFrame(MAC_r)
MAC_r = MAC_r.set_index(np.array(R_data_files))
MAC_r.columns = Lambda_values

babs = [((0.48*(ODs[:,x])**1.32)/Mass_r[x,6])*(np.pi*(d**2)/4)*1e6 for x in range(len(Mass_r))]
babs = np.asarray(babs)
babs = pd.DataFrame(babs)
babs = babs.set_index(np.array(R_data_files))
babs.columns = Lambda_values

#Saving MAC and babs Matrix to Excel ##FilterChangeReq
MAC_r.to_excel(excel_writer=r'C:\Users/renyuxuan/Desktop/Research/Black Carbon/Analysis-10May2023_Joshin\Results\MAC_ALL.xlsx')
babs.to_excel(excel_writer=r'C:\Users/renyuxuan/Desktop/Research/Black Carbon/Analysis-10May2023_Joshin\Results\Babs_ALL.xlsx')

######## Update2: By Joshin | Code to make a dashboard for data visulization
#Creating Analysis DataFrame
#Analysis = pd.DataFrame([Mass['Filter_ID'], Mass['Sampling Start Date'], Mass['Sampling End Date']])
Mass = Mass.set_index(np.array(R_data_files))
data = [Mass['Filter ID'], Mass['Sampling Start Date'], Mass['Sampling End Date'], Mass['PM2.5(ug/m3)'], babs[900], babs[653], babs[403], MAC_r[900], MAC_r[653], MAC_r[403]]
Analysis = pd.concat(data, axis=1)
Analysis = Analysis.set_index(np.array(R_data_files))
Analysis.columns = ['Filter ID', 'Sampling Start Date', 'Sampling End Date', 'PM2.5(ug/m3)', 'Abs900nm', 'Abs653nm', 'Abs403nm', 'MAC900nm', 'MAC653nm', 'MAC403nm']

Analysis['f_BC'] = (Analysis['Abs900nm']/Analysis['PM2.5(ug/m3)'])/4.58

Analysis['Mass_BC(ug/m3)'] = Analysis['f_BC']*Analysis['PM2.5(ug/m3)']

AAE_used = 0.9
Analysis['Res643nm'] = Analysis['Abs653nm'] - (Analysis['Abs900nm']*(900/653)**(AAE_used))
Analysis['Res403nm'] = Analysis['Abs403nm'] - (Analysis['Abs900nm']*(900/403)**(AAE_used))

Analysis['AAE(900,403)'] = -1*np.log(Analysis['Abs900nm']/Analysis['Abs403nm'])/np.log(900/403)
Analysis['AAE(900,653)'] = -1*np.log(Analysis['Abs900nm']/Analysis['Abs653nm'])/np.log(900/653)
Analysis['AAE(653,403)'] = -1*np.log(Analysis['Abs653nm']/Analysis['Abs403nm'])/np.log(653/403)

Analysis.drop(labels=blank_data_files, inplace=True)

Analysis['Location ID'] = Analysis['Filter ID'].str[:4]

import datetime
Analysis['Month'] = pd.to_datetime(Analysis['Sampling Start Date']).dt.strftime("%B")
Analysis['Year'] = pd.to_datetime(Analysis['Sampling Start Date']).dt.strftime("%Y")

Analysis['Sampling Start Date'] = pd.to_datetime(Analysis['Sampling Start Date']).dt.date
# Analysis['Sampling End Date'] = pd.to_datetime(Analysis['Sampling End Date']).dt.date

############ INSTEAD OF THE COMMENTED LINE ABOVE (To accomodate 29th Feb error-less-ly)
# convert 'Sampling End Date' column to datetime type
Analysis['Sampling End Date'] = pd.to_datetime(Analysis['Sampling End Date'], errors='coerce')
# find the rows where the date is 29th Feb
mask = (Analysis['Sampling End Date'].dt.day == 29) & (Analysis['Sampling End Date'].dt.month == 2)
# add 1 day to these dates
Analysis.loc[mask, 'Sampling End Date'] += pd.offsets.Day(1)
# convert 'Sampling End Date' column to date type
Analysis['Sampling End Date'] = Analysis['Sampling End Date'].dt.date
############

Analysis.to_excel(excel_writer=r'C:\Users/renyuxuan/Desktop/Research/Black Carbon/Analysis-10May2023_Joshin\Results\Analysis_ALL.xlsx')
Analysis

# ############################# Fresh UV-VIS DATA ANALYSIS RUN CODE: 17th May 2023
# #@Author: Joshin Kumar

# #Define constants
# d = 25*1e-3


# import os
# import pandas as pd

# ########### READ UV-VIS TRANSMITTANCE FILES
# combined_df = pd.DataFrame()

# for filename in os.listdir(Transmittance_directory_path):
#     if filename.endswith(".csv"):
#         file_path = os.path.join(Transmittance_directory_path, filename)
#         df = pd.read_csv(file_path, delimiter=',')

#         nm_column_name = df.columns[0]
#         transmittance_column_name = df.columns[1]

#         df = df.set_index(nm_column_name)[transmittance_column_name].rename(filename)

#         combined_df = pd.concat([combined_df, df], axis=1)

# All_Transmittance_df = combined_df.transpose().copy()
# display(All_Transmittance_df)
# ########### READ UV-VIS REFLECTANCE FILES
# combined_df = pd.DataFrame()

# for filename in os.listdir(Reflectance_directory_path):
#     if filename.endswith(".csv"):
#         file_path = os.path.join(Reflectance_directory_path, filename)
#         df = pd.read_csv(file_path, delimiter=',')

#         nm_column_name = df.columns[0]
#         transmittance_column_name = df.columns[1]

#         df = df.set_index(nm_column_name)[transmittance_column_name].rename(filename)

#         combined_df = pd.concat([combined_df, df], axis=1)

# All_Reflectance_df = combined_df.transpose().copy()
# display(All_Reflectance_df)