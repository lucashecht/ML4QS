import math
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.decomposition import PCA
from ML4QS.Python3Code.Chapter4.FrequencyAbstraction import FourierTransformation
import re
import os
from statistics import mean

import xml.etree.ElementTree as ET
import re
from datetime import datetime

def readtcx(path):
    heartrate_data = []
    with open(path) as xml_file:
        xml_str = xml_file.read()
        xml_str = re.sub(' xmlns="[^"]+"', '', xml_str, count=1)
        root = ET.fromstring(xml_str)
        activities = root.findall('.//Activity')
        for activity in activities:
            tracking_points = activity.findall('.//Trackpoint')
            for tracking_point in list(tracking_points):
                children = list(tracking_point)
                time = datetime.strptime(children[0].text, '%Y-%m-%dT%H:%M:%S.%fZ')
                hr = list(tracking_point.find('HeartRateBpm'))[0].text
                heartrate_data.append([time, hr])
    df = pd.DataFrame(heartrate_data, columns=['time', 'hr'])
    #df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def time_diff(df):
    res = []
    for i in range(len(df["Time (s)"]) - 1):
        val1 = df["Time (s)"][i]
        val2 = df["Time (s)"][i + 1]
        
        res.append(val2 - val1)
    return mean(res)

def read_phyphox(parent_dir):
    acc = pd.read_csv(os.path.join(parent_dir, "Accelerometer.csv"))
    gyro = pd.read_csv(os.path.join(parent_dir, "Gyroscope.csv"))
    loc = pd.read_csv(os.path.join(parent_dir, "Location.csv"))
        
    data_len = min(len(acc), len(gyro))
    acc = acc[0:data_len]
    gyro = gyro[0:data_len]
        
    time_step = mean([time_diff(acc), time_diff(gyro)])
    
    for i in range(data_len):
        time = time_step * i
        acc["Time (s)"][i] = time
        gyro["Time (s)"][i] = time
    
    acc.set_index('Time (s)', inplace=True)
    gyro.set_index('Time (s)', inplace=True)
    loc.set_index('Time (s)', inplace=True)
        
            
    merged = acc.join(gyro, how="outer")
    merged = pd.concat([merged, loc]).sort_index().interpolate()
    
    # Rename columns
    merged.index.names = ["time"]
    merged.rename(inplace=True, columns={
        "Acceleration x (m/s^2)": "acceleration_x",
        "Acceleration y (m/s^2)": "acceleration_y",
        "Acceleration z (m/s^2)": "acceleration_z",
        "Gyroscope x (rad/s)": "gyroscope_x",
        "Gyroscope y (rad/s)": "gyroscope_y",
        "Gyroscope z (rad/s)": "gyroscope_z",
        "Latitude (°)": "latitude",
        "Longitude (°)": "longitude",
        "Height (m)": "height",
        "Velocity (m/s)": "velocity",
        "Direction (°)": "direction",
        "Horizontal Accuracy (m)": "h_accuracy",
        "Vertical Accuracy (m)": "v_accuracy",

    })
    merged = merged.dropna()
    
    time_df = pd.read_csv(os.path.join(parent_dir, "meta", "time.csv"))
    start_time = time_df.loc[time_df["event"] == "START"]["system time"][0]
    
    merged.reset_index(inplace=True)
    merged['time'] = pd.to_datetime(merged['time'] + start_time,unit='s')
    return merged

def read_combined(path_pp, path_tcx):
    pp = read_phyphox(path_pp)
    hr = readtcx(path_tcx)

    pp["hr"] = np.nan


    # Finds the time intervals of the heart rate measurements and update the phyphox heart rate accordingly
    # This is done as samsung measurements are not very fine grained and only give use relatively large
    # time intervals
    for i in range(len(hr) - 1):
        row1 = hr.iloc[i]
        row2 = hr.iloc[i + 1]
        pp.loc[(pp["time"] >= row1["time"].to_datetime64()) & (pp["time"] < row2["time"].to_datetime64()), "hr"] = row1["hr"]

    # remove data point without overlap in time
    return pp.dropna()
df = read_combined("../data/Lucas/walking2/", "../data/Lucas/walking2/walking2.tcx")
np.random.seed(0)

fs = 4

periodic_predictor_cols = ['acceleration_x', 'acceleration_y', 'acceleration_z', 'gyroscope_x', 'gyroscope_y', 'gyroscope_z']

FreqAbs = FourierTransformation()
data_table = FreqAbs.abstract_frequency(copy.deepcopy(df), periodic_predictor_cols[3:], 40, fs)

# Get the frequencies from the columns....
frequencies = []
values = []
for col in data_table.columns:
    val = re.findall(r'freq_\d+\.\d+_Hz', col)
    if len(val) > 0:
        frequency = float((val[0])[5:len(val)-4])
        frequencies.append(frequency)
        values.append(data_table.loc[data_table.index, col])

fig = plt.figure()
ax1 = fig.add_subplot(111)
#plt.xlim([0, 5])
ax1.plot(frequencies, values, 'b+')
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('$a$')
plt.draw()

plt.figure(figsize=(15,5))
plt.plot(data_table['time'], data_table['gyroscope_x_max_freq'])
plt.draw()
plt.figure(figsize=(15,5))
plt.plot(data_table['time'], data_table['gyroscope_x_freq_weighted'])
plt.ylim([-3,10])
plt.draw()
plt.figure(figsize=(15,5))
plt.plot(data_table['time'], data_table['gyroscope_x_pse'])
plt.show()
