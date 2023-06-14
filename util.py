from statistics import mean
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os, re
import xml.etree.ElementTree as ET
from datetime import datetime


def time_diff(df):
    res = []
    for i in range(len(df["Time (s)"]) - 1):
        val1 = df["Time (s)"][i]
        val2 = df["Time (s)"][i + 1]
        
        res.append(val2 - val1)
    return mean(res)

def read_phyphox(parent_dir):
    acc = pd.read_csv(os.path.join(parent_dir, "Phyphox", "Accelerometer.csv"))
    gyro = pd.read_csv(os.path.join(parent_dir, "Phyphox", "Gyroscope.csv"))
    loc = pd.read_csv(os.path.join(parent_dir, "Phyphox", "Location.csv"))
        
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
    
    time_df = pd.read_csv(os.path.join(parent_dir, "Phyphox", "meta", "time.csv"))
    start_time = time_df.loc[time_df["event"] == "START"]["system time"][0]
    
    merged.reset_index(inplace=True)
    merged['time'] = pd.to_datetime(merged['time'] + start_time, unit='s')
    return merged



def readtcx(parent_dir):
    # Function to parse heart rate data from .tcx files my Garmin produces
    # Returns DataFrame with 'time' and 'hr' columns
    
    heartrate_data = []    
    # iterating over all files
    for files in os.listdir(parent_dir):
        if files.endswith("tcx"):
            print(files)
            with open(os.path.join(parent_dir, files)) as xml_file:
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
        else:
            continue
    
    df = pd.DataFrame(heartrate_data, columns=['start_time', 'heart_rate']).dropna()
    return df


def read_samsung_health(parent_dir):
    files = os.scandir(os.path.join(parent_dir, "SamsungHealth"))
    filename = next(filter(lambda file: "com.samsung.health.exercise.live_data.json" in file.name, files)).name
        
    df = pd.read_json(os.path.join(parent_dir, "SamsungHealth", filename))
    df = df[["start_time", "heart_rate"]].dropna()

    return df

def read_combined(parent_dir, activity_type, file_type):
    pp = read_phyphox(parent_dir)

    # if samsung data or tcx format for heart_rate
    if file_type == "json":
        sh = read_samsung_health(parent_dir)
    else:
        sh = readtcx(parent_dir)

    pp["heart_rate"] = np.nan
    
    # Finds the time intervals of the heart rate measurements and update the phyphox heart rate accordingly
    # This is done as samsung measurements are not very fine grained and only give use relatively large
    # time intervals
    for i in range(len(sh) - 1):
        row1 = sh.iloc[i]
        row2 = sh.iloc[i + 1]
        pp.loc[(pp["time"] >= row1["start_time"]) & (pp["time"] < row2["start_time"]), "heart_rate"] = row1["heart_rate"]
    df = pp.dropna()

    # add an extra column to add the activity type
    df["label"] = activity_type
    return df

