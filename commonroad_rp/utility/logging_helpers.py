import sys
import os
import numpy as np
import logging
import json
from pathlib import Path
from enum import Enum

from commonroad_rp.trajectories import TrajectorySample


class LogMode(Enum):
    visualization = 2
    evaluation = 1
    none = 3


class DataLoggingCosts:
    # ----------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    def __init__(
        self, path_logs: str, header_only: bool = False, save_all_traj: bool = False, log_mode: int = 1
    ) -> None:
        """"""

        self.save_all_traj = save_all_traj

        self.log_mode = LogMode(log_mode)

        self.header = None
        self.trajectories_header = None
        self.prediction_header = None
        self.path_logs = path_logs

        log_file_name = "logs.csv"
        prediction_file_name = "predictions.csv"
        collision_file_name = "collision.csv"
        self.trajectories_file_name = "trajectories.csv"
        if header_only:
            return
        self.trajectory_number = 0

        self.__trajectories_log_path = None

        # Create directories
        if not os.path.exists(path_logs):
            os.makedirs(path_logs)
        self.__log_path = os.path.join(path_logs, log_file_name)
        self.__prediction_log_path = os.path.join(
            path_logs, prediction_file_name)
        self.__collision_log_path = os.path.join(
            path_logs, collision_file_name)
        Path(os.path.dirname(self.__log_path)).mkdir(
            parents=True, exist_ok=True)

        self.set_logging_header()

    # ----------------------------------------------------------------------------------------------------------
    # CLASS METHODS --------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    def set_logging_header(self, cost_function_names=None):

        cost_names = str()
        if cost_function_names:
            for names in cost_function_names.keys():
                cost_names += cost_function_names[names].__name__ + ";"

        self.header = (
            "trajectory_number;"
            "calculation_time_s;"
            "infeasible_kinematics_sum;"
            "inf_kin_acceleration;"
            "inf_kin_negative_s_velocity;"
            "inf_kin_max_s_idx;"
            "inf_kin_negative_v_velocity;"
            "inf_kin_max_curvature;"
            "inf_kin_yaw_rate;"
            "inf_kin_max_curvature_rate;"
            "inf_kin_vehicle_acc;"
            "infeasible_collision;"
            "x_positions_m;"
            "y_positions_m;"
            "theta_orientations_rad;"
            "velocities_mps;"
            "accelerations_mps2;"
            "s_position_m;"
            "d_position_m;"
            "cluster_number;"
            "costs_cumulative_weighted;"
            +
            cost_names
            +
            "prediction_cost;"
            "responsibility_cost;"
        )
        self.trajectories_header = (
            "time_step;"
            "trajectory_number;"
            "unique_id;"
            "feasible;"
            "horizon;"
            "dt;"
            "x_positions_m;"
            "y_positions_m;"
            "theta_orientations_rad;"
            "velocities_mps;"
            "accelerations_mps2;"
            "s_position_m;"
            "d_position_m;"
            "cluster_number;"
            "costs_cumulative_weighted;"
            +
            cost_names
            +
            "prediction_cost;"
            "responsibility_cost;"
        )

        self.prediction_header = (
            "trajectory_number;"
            "prediction"
        )

        # write header to logging file
        with open(self.__log_path, "w+") as fh:
            fh.write(self.header)

        with open(self.__prediction_log_path, "w+") as fh:
            fh.write(self.prediction_header)

        if self.save_all_traj:
            self.__trajectories_log_path = os.path.join(
                self.path_logs, self.trajectories_file_name)
            with open(self.__trajectories_log_path, "w+") as fh:
                fh.write(self.trajectories_header)

    def get_headers(self):
        return self.header

    # def log_cost(
    #     self,
    #     costs: list
    # ) -> None:
    #     """log_data _summary_
    #     """
    #     if (self.log_mode == LogMode.visualization or self.log_mode == LogMode.evaluation):
    #         with open(self.__log_path, "a") as fh:
    #             fh.write(
    #                 "\n"
    #                 + str(self.trajectory_number)
    #                 + ";"
    #                 + json.dumps(str(costs[0]))
    #                 + ";"
    #                 + json.dumps(str(costs[1]))
    #                 + ";"
    #                 + json.dumps(str(costs[2]))
    #                 + ";"
    #                 + json.dumps(str(costs[3]))
    #                 + ";"
    #                 + json.dumps(str(costs[4]))
    #                 + ";"
    #                 + json.dumps(str(costs[5]))
    #                 + ";"
    #                 + json.dumps(str(costs[6]))
    #                 + ";"
    #                 + json.dumps(str(costs[7]))
    #                 + ";"
    #                 + json.dumps(str(costs[8]))
    #                 + ";"
    #                 + json.dumps(str(costs[9]))
    #                 + ";"
    #                 + json.dumps(str(costs[10]))
    #                 + ";"
    #                 + json.dumps(str(costs[11]))
    #             )

    def log(self, trajectory: TrajectorySample, infeasible_kinematics, infeasible_collision: int, planning_time: float, cluster: int,
            collision: bool = False):
        if (self.log_mode == LogMode.visualization or self.log_mode == LogMode.evaluation):
            new_line = "\n" + str(self.trajectory_number)

            cartesian = trajectory._cartesian
            cost_list = trajectory._cost_list

            # log time
            new_line += ";" + json.dumps(str(planning_time), default=default)

            # log infeasible
            for kin in infeasible_kinematics:
                new_line += ";" + json.dumps(str(kin), default=default)
            new_line += ";" + \
                json.dumps(str(infeasible_collision), default=default)

            # log position
            new_line += ";" + json.dumps(str(','.join(map(str, cartesian.x))), default=default)
            new_line += ";" + json.dumps(str(','.join(map(str, cartesian.y))), default=default)
            new_line += ";" + json.dumps(str(','.join(map(str, cartesian.theta))), default=default)
            # log velocity & acceleration
            new_line += ";" + json.dumps(str(','.join(map(str, cartesian.v))), default=default)
            new_line += ";" + json.dumps(str(','.join(map(str, cartesian.a))), default=default)

            # # log frenet coordinates (distance to reference path)
            new_line += ";" + \
                json.dumps(str(trajectory._curvilinear.s[0]), default=default)
            new_line += ";" + \
                json.dumps(str(trajectory._curvilinear.d[0]), default=default)

            # log cluster number
            new_line += ";" + json.dumps(str(cluster), default=default)

            # log costs
            new_line += ";" + json.dumps(str(trajectory._cost), default=default)
            # log costs
            for cost in cost_list:
                new_line += ";" + json.dumps(str(cost), default=default)

            with open(self.__log_path, "a") as fh:
                fh.write(new_line)

    def log_pred(self, prediction):
        if (self.log_mode == LogMode.visualization or self.log_mode == LogMode.evaluation):
            new_line = "\n" + str(self.trajectory_number)

            new_line += ";" + json.dumps(prediction, default=default)

            with open(self.__prediction_log_path, "a") as fh:
                fh.write(new_line)

    def log_collision(self, collision_with_obj, ego_length, ego_width, center=None, last_center=None, r_x=None, r_y=None, orientation=None):
        self.collision_header = (
            "ego_length;"
            "ego_width;"
            "center_x;"
            "center_y;"
            "last_center_x;"
            "last_center_y;"
            "r_x;"
            "r_y;"
            "orientation"
        )
        if (self.log_mode == LogMode.evaluation):

            with open(self.__collision_log_path, "w+") as fh:
                fh.write(self.collision_header)

            new_line = "\n" + str(ego_length)
            new_line += ";" + str(ego_width)
            if collision_with_obj:
                new_line += ";" + str(center[0])
                new_line += ";" + str(center[1])
                new_line += ";" + str(last_center[0])
                new_line += ";" + str(last_center[1])
                new_line += ";" + str(r_x)
                new_line += ";" + str(r_y)
                new_line += ";" + str(orientation)
            else:
                new_line += ";None;None;None;None;None;None;None"

            with open(self.__collision_log_path, "a") as fh:
                fh.write(new_line)

    def log_all_trajectories(self, all_trajectories, time_step, cluster: int):
        if (self.log_mode == LogMode.visualization):
            i = 0
            for trajectory in all_trajectories:
                self.log_trajectory(trajectory, i, time_step, trajectory.valid, cluster)
                i += 1

    def log_trajectory(self, trajectory: TrajectorySample, trajectory_number: int, time_step, feasible: bool, cluster: int):
        new_line = "\n" + str(time_step)
        new_line += ";" + str(trajectory_number)
        new_line += ";" + str(trajectory._unique_id)
        new_line += ";" + str(feasible)
        new_line += ";" + str(trajectory.horizon)
        new_line += ";" + str(trajectory.dt)

        cartesian = trajectory._cartesian
        cost_list = trajectory._cost_list

        # log position
        new_line += ";" + json.dumps(str(','.join(map(str, cartesian.x))), default=default)
        new_line += ";" + json.dumps(str(','.join(map(str, cartesian.y))), default=default)
        new_line += ";" + json.dumps(str(','.join(map(str, cartesian.theta))), default=default)
        # log velocity & acceleration
        new_line += ";" + json.dumps(str(','.join(map(str, cartesian.v))), default=default)
        new_line += ";" + json.dumps(str(','.join(map(str, cartesian.a))), default=default)

        # log frenet coordinates (distance to reference path)
        new_line += ";" + \
            json.dumps(str(trajectory._curvilinear.s[0]), default=default)
        new_line += ";" + \
            json.dumps(str(trajectory._curvilinear.d[0]), default=default)

        # log x, y, yaw
        # new_line += ";" + \
        #     json.dumps(trajectory.x, default=default)
        # new_line += ";" + \
        #     json.dumps(trajectory.y, default=default)
        # new_line += ";" + \
        #     json.dumps(trajectory.yaw, default=default)

        # log _trajectory_long and _trajectory_lat
        # new_line += ";" + \
        #     json.dumps(trajectory._trajectory_long.__dict__, default=default)
        # new_line += ";" + \
        #     json.dumps(trajectory._trajectory_lat.__dict__, default=default)

        # log cluster number
        new_line += ";" + json.dumps(str(cluster), default=default)

        # log costs
        new_line += ";" + json.dumps(str(trajectory._cost), default=default)
        for cost in cost_list:
            new_line += ";" + json.dumps(str(cost), default=default)

        with open(self.__trajectories_log_path, "a") as fh:
            fh.write(new_line)


def default(obj):
    # handle numpy arrays when converting to json
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError("Not serializable (type: " + str(type(obj)) + ")")
