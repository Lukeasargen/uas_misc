import os
import glob

import pandas as pd
import numpy as np
from scipy import stats


def extract_data(log_dir, log_filename):

    csv_filename = lambda k: os.path.join(log_dir, log_filename, k+".csv")
    
    with open(log_dir+log_filename+".log", "r") as filehandle:
        log_list = filehandle.readlines()
    os.makedirs(log_dir+log_filename, exist_ok=True)

    # First find all the FMT messages and make empty arrays
    fmt_list = [i for i in log_list if "FMT," in i]
    save_data = {}
    print("Number of fmt:", len(fmt_list))
    for line in fmt_list:
        s = line.split(", ")
        save_data[s[3]] = [s[-1]]

    # Go through each line and make add each csv line
    for line in log_list:
        # if "FMT," not in line:
        idx = line.find(',')
        save_data[line[:idx]].append(line[idx+2:])

    # Save everything
    for k in save_data.keys():
        with open(csv_filename(k), "w+") as f:
            f.writelines(save_data[k])

    # Load dataframes and convert to data types
    data = {}
    for csv_path in glob.glob(csv_filename("*")):
        k = os.path.splitext(os.path.basename(csv_path))[0]
        if k=="FMT" or k=="MSG": continue  # this has extra commas
        data[k] = pd.read_csv(csv_filename(k), delimiter=",")
        try:
            data[k] = data[k].apply(pd.to_numeric)
        except:
            print("No numerics:", k)
    
    data["PARM"]["Name"] = data["PARM"]["Name"].astype('string')
    data["MODE"]["Mode"] = data["MODE"]["Mode"].astype('string')
    return data


def remove_outlier_rows(df, col, z_max=3):
    while True:
        z_mag = np.abs(stats.zscore(df[col]))
        idxmax = z_mag.idxmax()
        if z_mag[idxmax]<z_max: break  # if good, break before drop
        df = df.drop(idxmax)
    return df.copy()


def find_closest_row(value, df, colname, use_next=False):
    """ Finds the closest previous row."""
    # Fastest way to find the neighboring rows
    next_row = np.searchsorted(df[colname].values, value)
    # If the next row is the first row, just return the first row
    if next_row==0:
        return df.iloc[next_row]
    # Finding the previous row
    prev_row = (next_row-1).clip(0)
    return df.iloc[next_row] if use_next else df.iloc[prev_row]


def slice_between(df, col, low, high):
    return df[df[col].between(low, high)].copy()

def get_param(df, name, time=0):
    param_df = df["PARM"][df["PARM"]["Name"]==" "+name]
    return float(find_closest_row(time, param_df, "TimeUS")["Value"])