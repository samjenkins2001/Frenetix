import os
import pandas as pd
import cv2
import glob
import numpy as np
import re


def lowest_costs(df):
    dff = df[df["variable"] == "costs_cumulative_weighted"]
    dff = dff.sort_values(by='value')
    dff = dff.reset_index().truncate(before=0, after=9)
    trajs = dff["trajectory_number"].unique()
    df = df[df["trajectory_number"].isin(trajs)]
    return df


def get_cost_headers_and_remaining(df):
    headers = df.columns.tolist()
    substring = "cost"                       # all attributes with the substring cost are assumed to be cost factors
    cost_headers = []
    for header in headers:
        if substring in header:
            cost_headers.append(header)
    remaining_headers = [x for x in headers if x not in cost_headers]
    return cost_headers, remaining_headers


def get_images(logs_path):
    images_list = os.path.join(logs_path, "plots", "*.png")
    images_list = glob.glob(images_list)
    final_plot = [s for s in images_list if "final" in s]
    if final_plot:
        images_list.remove(final_plot[0])
    images_list.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

    img = []
    for i in range(0, len(images_list)):
        img.append(cv2.imread(images_list[i]))
    img = np.asarray(img, dtype=np.uint8)
    return img


def setup_trajs(csv_name):
    df = pd.read_csv(csv_name, delimiter=";")
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # headers related to costs and remaining
    cost_headers, remaining_headers = get_cost_headers_and_remaining(df)

    # make cost headers to attribute -> csv becomes way longer
    df = pd.melt(df, id_vars=remaining_headers, value_vars=cost_headers, value_name='value')

    df['x_positions_m'] = df['x_positions_m'].apply(lambda x: x.split(","))
    df['x_positions_m'] = df.apply(lambda row: row.x_positions_m[row.actual_traj_length-1], axis=1)  # shorten to plottet trajectories

    df['y_positions_m'] = df['y_positions_m'].apply(lambda x: x.split(","))
    df['y_positions_m'] = df.apply(lambda row: row.y_positions_m[row.actual_traj_length-1], axis=1)  # shorten to plottet trajectories

    df['velocities_mps'] = df['velocities_mps'].apply(lambda x: float(x.split(",")[0]))  # only use initial velocity for each timestep
    df['accelerations_mps2'] = df['accelerations_mps2'].apply(lambda x: float(x.split(",")[0]))  # only use initial velocity for each timestep

    df = df.replace(True, 'feasible')
    df = df.replace(False, 'infeasible')

    df_two = df.copy(deep=True)
    df_two = df_two.assign(feasible='All')
    df = pd.concat([df, df_two])

    return df


def setup_logs(csv_name):
    logs = pd.read_csv(csv_name, delimiter=";")
    logs['x_positions_m'] = logs['x_positions_m'].apply(lambda x: x.split(",")[0])
    logs['y_positions_m'] = logs['y_positions_m'].apply(lambda x: x.split(",")[0])
    logs['velocities_mps'] = logs['velocities_mps'].apply(lambda x: x.split(",")[0])
    logs['accelerations_mps2'] = logs['accelerations_mps2'].apply(lambda x: x.split(",")[0])
    logs['theta_orientations_rad'] = logs['theta_orientations_rad'].apply(lambda x: x.split(",")[0])
    return logs


def get_latest_log_files(repo_path) -> str:

    try:
        latest_loc_dir = max(glob.glob(os.path.join(repo_path, '*/')), key=os.path.getmtime)
    except Exception as e:
        raise IOError(e, "\n\nNo logs available, check output or run new simulation!")

    if os.path.isfile(os.path.join(latest_loc_dir, "logs.csv")):
        pass
    else:
        try:
            latest_loc_dir = max(glob.glob(os.path.join(latest_loc_dir, '*/')), key=os.path.getmtime)
        except:
            raise IOError("No valid log file format!")

    return latest_loc_dir



