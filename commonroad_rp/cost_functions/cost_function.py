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
from omegaconf import OmegaConf
import commonroad_rp.trajectories
import commonroad_rp.cost_functions.partial_cost_functions as cost_functions

from risk_assessment.collision_probability import (
    get_collision_probability_fast
)
from risk_assessment.risk_costs import get_responsibility_cost
from risk_assessment.risk_costs import calc_risk


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
    #T = "T"
    #ID = "ID"
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

    def __init__(self, rp, cost_function_params, save_unweighted_costs):
        super(AdaptableCostFunction, self).__init__()

        self.scenario = None
        self.rp = None
        self.reachset = None
        self.desired_speed = None
        self.predictions = None

        self.vehicle_params = rp.vehicle_params
        self.params = OmegaConf.to_object(cost_function_params)
        self.save_unweighted_costs = save_unweighted_costs

        self.PartialCostFunctionMapping = {
            PartialCostFunction.A: cost_functions.acceleration_cost,
            PartialCostFunction.J: cost_functions.jerk_cost,
            PartialCostFunction.Jlat: cost_functions.jerk_lat_cost,
            PartialCostFunction.Jlon: cost_functions.jerk_lon_cost,
            PartialCostFunction.O: cost_functions.orientation_offset_cost,
            PartialCostFunction.L: cost_functions.path_length_cost,

            PartialCostFunction.LC: cost_functions.lane_center_offset_cost,
            PartialCostFunction.V: cost_functions.velocity_offset_cost,
            PartialCostFunction.VC: cost_functions.velocity_costs,
            PartialCostFunction.DR: cost_functions.distance_to_reference_path_cost,
            PartialCostFunction.DO: cost_functions.distance_to_obstacles_cost,
            # PartialCostFunction.SA: cost_functions.steering_angle_cost,
            # PartialCostFunction.SR: cost_functions.steering_rate_cost,
            # PartialCostFunction.Y: cost_functions.yaw_cost,
            # PartialCostFunction.Vlon: cost_functions.longitudinal_velocity_offset_cost,
            # PartialCostFunction.T: cost_functions.time_cost,
            # PartialCostFunction.ID: cost_functions.inverse_duration_cost,
        }

        irrelevant_costs = []
        for category in list(self.params.keys()):
            for costs in list(self.params[category].keys()):
                if self.params[category][costs] == 0:
                    irrelevant_costs.append(costs)

        for idx, name in enumerate(list(self.PartialCostFunctionMapping.keys())):
            if str(name).split('.')[1] in irrelevant_costs:
                del self.PartialCostFunctionMapping[name]

    def update_state(self, scenario, rp, predictions, reachset):
        self.scenario = scenario
        self.rp = rp
        self.reachset = reachset
        self.desired_speed = rp._desired_speed
        self.predictions = predictions

    def evaluate(self, trajectory: commonroad_rp.trajectories.TrajectorySample, scenario="intersection"):

        total_cost = 0.0
        costlist = np.zeros(len(self.PartialCostFunctionMapping) + 2)
        costlist_weighted = np.zeros(len(self.PartialCostFunctionMapping) + 2)

        for num, function in enumerate(self.PartialCostFunctionMapping):
            costlist[num] = self.PartialCostFunctionMapping[function](trajectory, self.rp,
                                                                      self.scenario, self.desired_speed)
            costlist_weighted[num] = self.params[scenario][function.value] * costlist[num]

        if self.predictions is not None:

            if self.reachset is not None:
                ego_risk_dict, obst_risk_dict, ego_harm_dict, obst_harm_dict = calc_risk(
                    traj=trajectory,
                    ego_state=self.rp.x_0,
                    predictions=self.predictions,
                    scenario=self.scenario,
                    ego_id=24,
                    vehicle_params=self.vehicle_params,
                    road_boundary=self.rp.road_boundary,
                    params_harm=self.rp.params_harm,
                    params_risk=self.rp.params_risk,
                )
                responsibility_cost, bool_contain_cache = get_responsibility_cost(
                    scenario=scenario,
                    traj=trajectory,
                    ego_state=self.rp.x_0,
                    obst_risk_max=obst_risk_dict,
                    predictions=self.predictions,
                    reach_set=self.rp.reach_set
                )
            else:
                responsibility_cost = 0.0

            # prediction_costs_raw = get_mahalanobis_dist_dict(
            #     traj=trajectory,
            #     predictions=self.predictions,
            #     vehicle_params=self.vehicle_params
            # )

            prediction_costs_raw = get_collision_probability_fast(
                traj=trajectory,
                predictions=self.predictions,
                vehicle_params=self.vehicle_params
            )
            prediction_costs = 0
            for key in prediction_costs_raw:
                prediction_costs += np.sum(prediction_costs_raw[key])

            costlist[-2] = prediction_costs
            costlist_weighted[-2] = prediction_costs * self.params[scenario]["P"]

            if responsibility_cost * self.params[scenario]["R"] > prediction_costs * self.params[scenario]["P"]:
                costlist_weighted[-1] = costlist_weighted[-2] * 0.8
            else:
                costlist_weighted[-1] = responsibility_cost * self.params[scenario]["R"]

            costlist[-1] = responsibility_cost

            total_cost = np.sum(costlist_weighted)

        # Logging of Cost terms
        # self.costs_logger.trajectory_number = self.x_0.time_step
        # self.costs_logger.log_data(cost)

        if self.save_unweighted_costs:
            return total_cost, costlist
        else:
            return total_cost, costlist_weighted

