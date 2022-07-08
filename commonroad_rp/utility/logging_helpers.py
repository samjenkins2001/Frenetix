import sys
import os
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
            "time_costs_cost;"
            "inverse_duration_cost;"
            "velocity_cost;"
            "dist_obj_cost;"
            "prediction_cost"
        )
        file_name = "costs_logs.csv"
        if header_only:
            return
        self.trajectory_number = 0
        # Create directories
        if not os.path.exists(path_logs):
            os.makedirs(path_logs)
        self.__log_path = os.path.join(path_logs, file_name)
        Path(os.path.dirname(self.__log_path)).mkdir(parents=True, exist_ok=True)

        # write header to logging file
        with open(self.__log_path, "w+") as fh:
            fh.write(self.header)

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

    def log(self, trajectory: TrajectorySample, collision: bool = False):
        new_line = "\n" + str(self.trajectory_number)

        cartesian = trajectory._cartesian
        cost_list = trajectory._cost_list

        # log position
        new_line += ";" + str(cartesian.x[0])
        new_line += ";" + str(cartesian.y[0])

        # log velocity & acceleration
        new_line += ";" + str(cartesian.v[0])
        new_line += ";" + str(cartesian.a[0])

        # log frenet coordinates (distance to reference path)
        new_line += ";" + str(trajectory._curvilinear.s[0])
        new_line += ";" + str(trajectory._curvilinear.d[0])


        # log costs
        for cost in cost_list:
            new_line += ";" + str(cost)

        with open(self.__log_path, "a") as fh:
            fh.write(new_line)
