import sys
import os
import numpy as np
import logging
import json
from pathlib import Path

from commonroad_rp.trajectories import TrajectorySample, CartesianSample


class DataLoggingCosts:
    # ----------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    def __init__(
        self, path_logs: str, header_only: bool = False, save_all_traj: bool = False
    ) -> None:
        """"""

        self.save_all_traj = save_all_traj

        self.header = (
            "trajectory_number;"
            "planning_time;"
            "infeasible_kinematics;"
            "infeasible_collision;"
            "x_position;"
            "y_position;"
            "velocity;"
            "acceleration;"
            "s_position;"
            "d_position;"
            "acceleration_cost;"
            "jerk_cost;"
            "jerk_lat_cost;"
            "jerk_long_cost;"
            "orientation_cost;"
            "path_length_cost;"
            "lane_center_offset_cost;"
            "velocity_offset_cost;"
            "velocity_cost;"
            "distance_to_reference_path_cost;"
            "distance_to_obstacles_cost;"
            "prediction_cost;"
        )
        self.prediction_header = (
            "trajectory_number;"
            "prediction"
        )
        log_file_name = "logs.csv"
        prediction_file_name = "predictions.csv"
        if header_only:
            return
        self.trajectory_number = 0
        # Create directories
        if not os.path.exists(path_logs):
            os.makedirs(path_logs)
        self.__log_path = os.path.join(path_logs, log_file_name)
        self.__prediction_log_path = os.path.join(path_logs, prediction_file_name)
        Path(os.path.dirname(self.__log_path)).mkdir(parents=True, exist_ok=True)

        # write header to logging file
        with open(self.__log_path, "w+") as fh:
            fh.write(self.header)

        with open(self.__prediction_log_path, "w+") as fh:
            fh.write(self.prediction_header)

    # ----------------------------------------------------------------------------------------------------------
    # CLASS METHODS --------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------

    def get_headers(self):
        return self.header

    def log_cost(
        self,
        costs: list
    ) -> None:
        """log_data _summary_
        """
        with open(self.__log_path, "a") as fh:
            fh.write(
                "\n"
                + str(self.trajectory_number)
                + ";"
                + json.dumps(str(costs[0]))
                + ";"
                + json.dumps(str(costs[1]))
                + ";"
                + json.dumps(str(costs[2]))
                + ";"
                + json.dumps(str(costs[3]))
                + ";"
                + json.dumps(str(costs[4]))
                + ";"
                + json.dumps(str(costs[5]))
                + ";"
                + json.dumps(str(costs[6]))
                + ";"
                + json.dumps(str(costs[7]))
                + ";"
                + json.dumps(str(costs[8]))
                + ";"
                + json.dumps(str(costs[9]))
                + ";"
                + json.dumps(str(costs[10]))
            )

    def log(self, trajectory: TrajectorySample, infeasible_kinematics: int, infeasible_collision: int, planning_time: float,
            collision: bool = False):
        new_line = "\n" + str(self.trajectory_number)

        cartesian = trajectory._cartesian
        cost_list = trajectory._cost_list

        # log time
        new_line += ";" + json.dumps(str(planning_time), default=default)

        # log infeasible
        new_line += ";" + json.dumps(str(infeasible_kinematics), default=default)
        new_line += ";" + json.dumps(str(infeasible_collision), default=default)

        # log position
        new_line += ";" + json.dumps(str(cartesian.x[0]), default=default)
        new_line += ";" + json.dumps(str(cartesian.y[0]), default=default)

        # log velocity & acceleration
        new_line += ";" + json.dumps(str(cartesian.v[0]), default=default)
        new_line += ";" + json.dumps(str(cartesian.a[0]), default=default)

        # log frenet coordinates (distance to reference path)
        new_line += ";" + json.dumps(str(trajectory._curvilinear.s[0]), default=default)
        new_line += ";" + json.dumps(str(trajectory._curvilinear.d[0]), default=default)


        # log costs
        for cost in cost_list:
            new_line += ";" + json.dumps(str(cost), default=default)

        with open(self.__log_path, "a") as fh:
            fh.write(new_line)

    def log_pred(self, prediction):
        new_line = "\n" + str(self.trajectory_number)

        new_line += ";" + json.dumps(prediction, default=default)

        with open(self.__prediction_log_path, "a") as fh:
            fh.write(new_line)


def default(obj):
    # handle numpy arrays when converting to json
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError("Not serializable (type: " + str(type(obj)) + ")")


