__author__ = "Alexander Hobmeier"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = []
__version__ = ""
__maintainer__ = "Alexander Hobmeier"
__email__ = "commonroad@lists.lrz.de"
__status__ = ""

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import yaml
from pathlib import Path
import commonroad_rp.trajectories
import commonroad_rp.commonroad_sa.partial_cost_functions as cost_functions

from Prediction.walenet.risk_assessment.collision_probability import (
    get_mahalanobis_dist_dict,
    ignore_vehicles_in_cone_angle
)


class PartialCostFunction(Enum):
    """
    See https://gitlab.lrz.de/tum-cps/commonroad-cost-functions/-/blob/master/costFunctions_commonRoad.pdf for more
    details.

    A: Acceleration,
    J: Jerk,
    Jlat: Lateral Jerk,
    Jlon: Longitudinal Jerk,
    SA: Steering Angle,
    SR: Steering Rate,
    Y: Yaw Rate,
    LC: Lane Center Offset,
    V: Velocity Offset,
    Vlon: Longitudinal Velocity Offset,
    O: Orientation Offset,
    DR: Distance to Reference Path,
    DO: Distance to Obstacles,
    L: Path Length,
    T: Time,
    ID: Inverse Duration,
    P: Prediction
    """
    A = "A"
    J = "J"
    Jlat = "Jlat"
    Jlon = "Jlon"
    #SA = "SA"
    #SR = "SR"
    #Y = "Y"
    LC = "LC"
    V = "V"
    VC = "VC"
    #Vlon = "Vlon"
    O = "O"
    DR = "DR"
    DO = "DO"
    L = "L"
    T = "T"
    ID = "ID"
    P = "P"


class CostFunction(ABC):
    """
    Abstract base class for new cost functions
    """

    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, trajectory: commonroad_rp.trajectories.TrajectorySample) -> float:
        """
        Computes the costs of a given trajectory sample
        :param trajectory: The trajectory sample for the cost computation
        :return: The cost of the given trajectory sample
        """
        pass


class AdaptableCostFunction(CostFunction):
    """
    Default cost function for comfort driving
    """

    def __init__(self, rp, predictions, timestep, scenario):
        super(AdaptableCostFunction, self).__init__()
        self.desired_speed = rp._desired_speed
        # self.desired_d = desired_d
        self.predictions = predictions
        self.vehicle_params = rp.vehicle_params
        self.walenet_cost_factor = rp.walenet_cost_factor
        #self.costs_logger = rp.costs_logger
        self.timestep = timestep
        self.scenario = scenario
        self.rp = rp

        path = Path.cwd().joinpath("configurations/cost_weights.yaml")
        if path.is_file():
            with path.open() as file:
                self.params = yaml.load(file, Loader=yaml.FullLoader)
        else:
            raise FileNotFoundError

    def evaluate(self, trajectory: commonroad_rp.trajectories.TrajectorySample, scenario="intersection"):

        PartialCostFunctionMapping = {
            PartialCostFunction.A: cost_functions.acceleration_cost,
            PartialCostFunction.J: cost_functions.jerk_cost,
            PartialCostFunction.Jlat: cost_functions.jerk_lat_cost,
            PartialCostFunction.Jlon: cost_functions.jerk_lon_cost,
            # PartialCostFunction.SA: cost_functions.steering_angle_cost,
            # PartialCostFunction.SR: cost_functions.steering_rate_cost,
            # PartialCostFunction.Y: cost_functions.yaw_cost,
            # PartialCostFunction.LC: cost_functions.lane_center_offset_cost,
            # PartialCostFunction.V: cost_functions.velocity_offset_cost,
            # PartialCostFunction.VC: cost_functions.velocity_costs,
            # PartialCostFunction.Vlon: cost_functions.longitudinal_velocity_offset_cost,
            PartialCostFunction.O: cost_functions.orientation_offset_cost,
            #PartialCostFunction.DR: cost_functions.distance_to_reference_path_cost,
            # PartialCostFunction.DR: cost_functions.distance_to_obstacles_cost,
            PartialCostFunction.L: cost_functions.path_length_cost,
            #PartialCostFunction.T: cost_functions.time_cost,
            #PartialCostFunction.ID: cost_functions.inverse_duration_cost,
        }

        costlist = list()

        for function in PartialCostFunctionMapping:
            costlist.append(self.params[scenario][function.value] * PartialCostFunctionMapping[function](trajectory))

        costlist.append(cost_functions.velocity_offset_cost(trajectory, self.desired_speed, self.params[scenario]["V"]))
        costlist.append(cost_functions.distance_to_reference_path_cost(trajectory, self.params[scenario]["DR"]))
        costlist.append(cost_functions.velocity_costs(trajectory, self.rp, self.scenario) * self.params[scenario]["VC"])
        costlist.append(cost_functions.lane_center_offset_cost(trajectory, self.scenario.lanelet_network) * self.params[scenario]["LC"])

        if self.predictions is not None:
            mahalanobis_costs = get_mahalanobis_dist_dict(
                traj=trajectory,
                predictions=self.predictions,
                vehicle_params=self.vehicle_params
            )
            costlist.append(self.params[scenario]["P"] * mahalanobis_costs * self.walenet_cost_factor)

        costlist.append(self.params[scenario]["DO"] * cost_functions.distance_to_obstacles_cost(trajectory, self.timestep, self.scenario))

        # Logging of Cost terms
        # self.costs_logger.trajectory_number = self.x_0.time_step
        # self.costs_logger.log_data(cost)

        return sum(costlist), costlist
