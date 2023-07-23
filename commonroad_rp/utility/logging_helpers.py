import os
import sys
import numpy as np
import json
from pathlib import Path
import logging
from cr_scenario_handler.utils.configuration import Configuration


class DataLoggingCosts:
    # ----------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    def __init__(self, path_logs: str, header_only: bool = False, save_all_traj: bool = False, cost_params: dict = None) -> None:
        """"""

        self.save_all_traj = save_all_traj

        self.header = None
        self.trajectories_header = None
        self.prediction_header = None
        self.collision_header = None

        self.path_logs = path_logs
        self._cost_list_length = None
        self.cost_names_list = None

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

        self.set_logging_header(cost_params)

    # ----------------------------------------------------------------------------------------------------------
    # CLASS METHODS --------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    def set_logging_header(self, cost_function_names=None):

        cost_names = str()
        if cost_function_names:
            self.cost_names_list = list(cost_function_names.keys())
            self.cost_names_list.sort()
            self._cost_list_length = len(self.cost_names_list)
            for names in self.cost_names_list:
                cost_names += names.replace(" ", "_") + '_cost;'

        self.header = (
            "trajectory_number;"
            "calculation_time_s;"
            "x_position_vehicle_m;"
            "y_position_vehicle_m;"
            "optimal_trajectory;"
            "percentage_feasible_traj;"
            "infeasible_kinematics_sum;"
            "inf_kin_acceleration;"
            "inf_kin_negative_s_velocity;"
            "inf_kin_max_s_idx;"
            "inf_kin_negative_v_velocity;"
            "inf_kin_max_curvature;"
            "inf_kin_yaw_rate;"
            "inf_kin_max_curvature_rate;"
            "inf_kin_vehicle_acc;"
            "inf_cartesian_transform;"
            "infeasible_collision;"
            "x_positions_m;"
            "y_positions_m;"
            "theta_orientations_rad;"
            "velocities_mps;"
            "accelerations_mps2;"
            "s_position_m;"
            "d_position_m;"
            "ego_risk;"
            "obst_risk;"
            "costs_cumulative_weighted;"
            +
            cost_names.strip(";")
        )
        self.trajectories_header = (
            "time_step;"
            "trajectory_number;"
            "unique_id;"
            "feasible;"
            "horizon;"
            "dt;"
            "actual_traj_length;"
            "x_positions_m;"
            "y_positions_m;"
            "theta_orientations_rad;"
            "velocities_mps;"
            "accelerations_mps2;"
            "s_position_m;"
            "d_position_m;"
            "ego_risk;"
            "obst_risk;"
            "costs_cumulative_weighted;"
            +
            cost_names
            +
            "inf_kin_yaw_rate;"
            "inf_kin_acceleration;"
            "inf_kin_max_curvature;"
            # "inf_kin_max_curvature_rate;"
        ).strip(";")



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

    def log(self, trajectory, infeasible_kinematics, percentage_kinematics, infeasible_collision: int, planning_time: float,
            collision: bool = False, ego_vehicle=None):

        new_line = "\n" + str(self.trajectory_number)

        if trajectory is not None:

            cartesian = trajectory.cartesian
            cost_list_names = list(trajectory.costMap.keys())

            # log time
            new_line += ";" + json.dumps(str(planning_time), default=default)

            # Vehicle Occupancy Position
            new_line += ";" + json.dumps(str(ego_vehicle.initial_state.position[0]), default=default)
            new_line += ";" + json.dumps(str(ego_vehicle.initial_state.position[1]), default=default)

            # optimaltrajectory available
            new_line += ";True"
            # log infeasible
            if percentage_kinematics is not None:
                new_line += ";" + json.dumps(str(percentage_kinematics), default=default)
            else:
                new_line += ";"

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
                json.dumps(str(trajectory.curvilinear.s[0]), default=default)
            new_line += ";" + \
                json.dumps(str(trajectory.curvilinear.d[0]), default=default)

            # log risk values number
            if trajectory._ego_risk is not None and trajectory._obst_risk is not None:
                new_line += ";" + json.dumps(str(trajectory._ego_risk), default=default)
                new_line += ";" + json.dumps(str(trajectory._obst_risk), default=default)
            else:
                new_line += ";;"

            new_line = self.log_costs_of_single_trajectory(trajectory, new_line, cost_list_names)

        else:
            # log time
            new_line += ";" + json.dumps(str(planning_time), default=default)
            new_line += ";False"
            # log infeasible
            for kin in infeasible_kinematics:
                new_line += ";" + json.dumps(str(kin), default=default)
            new_line += ";" + \
                        json.dumps(str(infeasible_collision), default=default)

            # log position
            new_line += ";None"
            new_line += ";None"
            new_line += ";None"
            # log velocity & acceleration
            new_line += ";None"
            new_line += ";None"

            # # log frenet coordinates (distance to reference path)
            new_line += ";None"
            new_line += ";None"

            # log costs
            new_line += ";None"
            # log costs
            for i in range(0, self._cost_list_length):
                new_line += ";None"

        with open(self.__log_path, "a") as fh:
            fh.write(new_line)

    def log_predicition(self, prediction):
        new_line = "\n" + str(self.trajectory_number)

        new_line += ";" + json.dumps(prediction, default=default)

        with open(self.__prediction_log_path, "a") as fh:
            fh.write(new_line)

    def log_collision(self, collision_with_obj, ego_length, ego_width, progress, center=None, last_center=None, r_x=None, r_y=None, orientation=None):
        self.collision_header = (
            "ego_length;"
            "ego_width;"
            "progress;"
            "center_x;"
            "center_y;"
            "last_center_x;"
            "last_center_y;"
            "r_x;"
            "r_y;"
            "orientation"
        )

        with open(self.__collision_log_path, "w+") as fh:
            fh.write(self.collision_header)

        new_line = "\n" + str(ego_length)
        new_line += ";" + str(ego_width)
        new_line += ";" + str(progress)
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

    def log_all_trajectories(self, all_trajectories, time_step):
        i = 0
        for trajectory in all_trajectories:
            self.log_trajectory(trajectory, i, time_step, trajectory.feasible)
            i += 1

    def log_trajectory(self, trajectory, trajectory_number: int, time_step, feasible: bool):
        new_line = "\n" + str(time_step)
        new_line += ";" + str(trajectory_number)
        new_line += ";" + str(trajectory.uniqueId)
        new_line += ";" + str(feasible)
        new_line += ";" + str(round(trajectory.sampling_parameters[1], 3))
        new_line += ";" + str(trajectory.dt)

        cartesian = trajectory.cartesian
        cost_list_names = list(trajectory.costMap.keys())

        new_line += ";" + str(int(round(trajectory.sampling_parameters[1], 3)/trajectory.dt))
        # log position
        new_line += ";" + json.dumps(str(','.join(map(str, cartesian.x))), default=default)
        new_line += ";" + json.dumps(str(','.join(map(str, cartesian.y))), default=default)
        new_line += ";" + json.dumps(str(','.join(map(str, cartesian.theta))), default=default)
        # log velocity & acceleration
        new_line += ";" + json.dumps(str(','.join(map(str, cartesian.v))), default=default)
        new_line += ";" + json.dumps(str(','.join(map(str, cartesian.a))), default=default)

        # log frenet coordinates (distance to reference path)
        new_line += ";" + \
            json.dumps(str(trajectory.curvilinear.s[0]), default=default)
        new_line += ";" + \
            json.dumps(str(trajectory.curvilinear.d[0]), default=default)

        # log risk values number
        if trajectory._ego_risk is not None and trajectory._obst_risk is not None:
            new_line += ";" + json.dumps(str(trajectory._ego_risk), default=default)
            new_line += ";" + json.dumps(str(trajectory._obst_risk), default=default)
        else:
            new_line += ";;"

        new_line = self.log_costs_of_single_trajectory(trajectory, new_line, cost_list_names)

        for k, v in trajectory.feasabilityMap.items():
            new_line += ";" + \
                json.dumps(str(v), default=default)



        with open(self.__trajectories_log_path, "a") as fh:
            fh.write(new_line)

    def log_costs_of_single_trajectory(self, trajectory, new_line, cost_list_names):

        # log costs
        new_line += ";" + json.dumps(str(trajectory.cost), default=default)

        # log costs
        for cost_template in self.cost_names_list:
            if cost_template in cost_list_names:
                new_line += ";" + json.dumps(str(trajectory.costMap[cost_template][1]), default=default)
            else:
                new_line += ";" + json.dumps(str(0), default=default)

        return new_line

def default(obj):
    # handle numpy arrays when converting to json
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError("Not serializable (type: " + str(type(obj)) + ")")


def messages_logger_initialization(config: Configuration, log_path) -> logging.Logger:
    """
    Message Logger Initialization
    """

    # msg logger
    msg_logger = logging.getLogger("Message_logger")

    # Create directories
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # create file handler (outputs to file)
    path_log = os.path.join(log_path, "messages.log")
    file_handler = logging.FileHandler(path_log)

    # set logging levels
    loglevel = config.debug.msg_log_mode
    msg_logger.setLevel(loglevel)
    file_handler.setLevel(loglevel)

    # create log formatter
    # formatter = logging.Formatter('%(asctime)s\t%(filename)s\t\t%(funcName)s@%(lineno)d\t%(levelname)s\t%(message)s')
    log_formatter = logging.Formatter("%(levelname)-8s [%(asctime)s] --- %(message)s (%(filename)s:%(lineno)s)",
                                  "%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(log_formatter)

    # create stream handler (prints to stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(loglevel)

    # create stream formatter
    stream_formatter = logging.Formatter("%(levelname)-8s [%(filename)s]: %(message)s")
    stream_handler.setFormatter(stream_formatter)

    # add handlers
    msg_logger.addHandler(file_handler)
    msg_logger.addHandler(stream_handler)
    msg_logger.propagate = False

    return msg_logger
