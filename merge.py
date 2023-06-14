from util import read_combined
import pandas as pd


def merge_all_date(data_dir):

    res_df = pd.DataFrame()
    temp_df = pd.DataFrame()


    for f,act_type, file_type in data_dir:
        temp_df = read_combined(f,act_type, file_type)
        res_df = pd.concat([res_df, temp_df], axis=0)
    
    # sort the final data based on the timeline
    res_df.sort_values(by='time', inplace = True) 
    return res_df


if __name__ == "__main__":
    # Parameters:   
    #   1. folder name
    #   2. type of activity
    #   3. type of file [especially for heart rate - 
    #       two types possible, json or tcx]

    data_dir = [("Bike 07-06-23 8:52","biking","json"),
                ("Bike 07-06-23 14:27","biking","json"),
                ("Tram 08-06-23 11:05","public_transport","json"),
                ("Walk 08-06-23 10:41","walking","json"),
                ("walking1","walking","tcx"),
                ("sitting1","sitting","tcx"),
                ("sitting2","sitting","tcx")]

                # ("walking2","walking","tcx"),

    target_dir = "./data-set/"
    res_df = merge_all_date(data_dir=data_dir)
    res_df.to_csv(target_dir + "data.csv")

