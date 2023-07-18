__author__ = "Alexander Hobmeier"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = []
__version__ = ""
__maintainer__ = "Alexander Hobmeier"
__email__ = "commonroad@lists.lrz.de"
__status__ = ""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List

import numpy as np
from omegaconf import OmegaConf
from commonroad_rp.trajectories import TrajectorySample
import commonroad_rp.cost_functions.partial_cost_functions as cost_functions
from commonroad_rp.cost_functions.cluster_based_cost_functions import ClusterBasedCostFunction

from risk_assessment.collision_probability import (
    get_collision_probability_fast, get_inv_mahalanobis_dist
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
    R = "R"


class CostFunction(ABC):
    """
    Abstract base class for new cost functions
    """

    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, trajectories: List[TrajectorySample]):
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

    def __init__(self, rp, configuration):
        super(AdaptableCostFunction, self).__init__()

        self.scenario = None
        self.rp = rp
        self.reachset = None
        self.desired_speed = None
        self.predictions = None
        self.configuration = configuration

        self.cluster_mapping = None
        self.cluster_prediction = None

        self.vehicle_params = rp.vehicle_params
        self.cost_weights = OmegaConf.to_object(configuration.cost.cost_weights)
        self.save_unweighted_costs = configuration.debug.save_unweighted_costs

        self.PartialCostFunctionMapping = {
            PartialCostFunction.A: cost_functions.acceleration_cost,
            PartialCostFunction.J: cost_functions.jerk_cost,
            PartialCostFunction.Jlat: cost_functions.jerk_lat_cost,
            PartialCostFunction.Jlon: cost_functions.jerk_lon_cost,
            PartialCostFunction.O: cost_functions.orientation_offset_cost,
            PartialCostFunction.L: cost_functions.path_length_cost,

            PartialCostFunction.LC: cost_functions.lane_center_offset_cost,
            PartialCostFunction.V: cost_functions.velocity_offset_cost,

            # TODO: In long term, delete VC cost term, because Velocity Planner is calculating target speed needed
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
        self.delete_irrelevant_costs()

    def delete_irrelevant_costs(self):
        irrelevant_costs = []
        for category in list(self.cost_weights.keys()):
            if self.cost_weights[category] == 0:
                irrelevant_costs.append(category)

        for idx, name in enumerate(list(self.PartialCostFunctionMapping.keys())):
            if str(name).split('.')[1] in irrelevant_costs:
                del self.PartialCostFunctionMapping[name]

    def update_state(self, scenario, rp, predictions, reachset):
        self.scenario = scenario
        self.rp = rp
        self.reachset = reachset
        self.desired_speed = rp._desired_speed
        self.predictions = predictions

    def get_cluster_name_by_index(self, index: int) -> str:
        return self.cluster_mapping[index]

    # calculate all costs for all trajcetories
    def evaluate(self, trajectories: List[TrajectorySample], scenario="cluster0"):

        # calculate prediction cost
        prediction_cost_list, responsibility_cost_list = self.calc_prediction(trajectories)

        # get cluster
        if self.use_clusters:
            cluster = self.cluster_prediction.evaluate(trajectories, prediction_cost_list)
            scenario = self.get_cluster_name_by_index(cluster)

        # calculate total cost
        self.calc_cost(trajectories, prediction_cost_list, responsibility_cost_list, scenario)

    # calculate prediction costs for all trajectories
    def calc_prediction(self, trajectories: List[TrajectorySample]):
        prediction_cost_list = []
        responsibility_cost_list = []
        for trajectory in trajectories:
            if self.predictions is not None and self.reachset is not None:
                ego_risk_dict, obst_risk_dict, ego_harm_dict, obst_harm_dict, ego_risk, obst_risk = calc_risk(
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
                trajectory._ego_risk = ego_risk
                trajectory._obst_risk = obst_risk

                responsibility_cost, bool_contain_cache = get_responsibility_cost(
                    scenario=self.scenario,
                    traj=trajectory,
                    ego_state=self.rp.x_0,
                    obst_risk_max=obst_risk_dict,
                    predictions=self.predictions,
                    reach_set=self.rp.reach_set
                )
            else:
                responsibility_cost = 0.0
            # prediction_costs_raw = get_collision_probability_fast(
            #     traj=trajectory,
            #     predictions=self.predictions,
            #     vehicle_params=self.vehicle_params
            # )
            prediction_costs_raw = get_inv_mahalanobis_dist(traj=trajectory, predictions=self.predictions,
                                                            vehicle_params=self.vehicle_params)

            prediction_costs = 0
            for key in prediction_costs_raw:
                prediction_costs += np.sum(prediction_costs_raw[key])

            prediction_cost_list.append(prediction_costs)
            responsibility_cost_list.append(responsibility_cost)
        return prediction_cost_list, responsibility_cost_list

    # calculate all costs (except prediction) and weigh them
    def calc_cost(self,  trajectories: List[TrajectorySample], prediction_cost_list, responsibility_cost_list):

        for i, trajectory in enumerate(trajectories):
            costlist = np.zeros(len(self.PartialCostFunctionMapping) + 2)
            costlist_weighted = np.zeros(len(self.PartialCostFunctionMapping) + 2)

            for num, function in enumerate(self.PartialCostFunctionMapping):
                costlist[num] = self.PartialCostFunctionMapping[function](trajectory, self.rp,
                                                                          self.scenario, self.desired_speed)
                costlist_weighted[num] = self.cost_weights[function.value] * costlist[num]

            # add prediction lost to list
            costlist[-2] = prediction_cost_list[i]
            costlist_weighted[-2] = prediction_cost_list[i] * self.cost_weights["P"]

            if any(responsibility_cost_list) > 0:
                if responsibility_cost_list[i] * self.cost_weights["R"] > prediction_cost_list[i] * self.cost_weights["P"]:
                    costlist_weighted[-1] = costlist_weighted[-2] * 0.8
                else:
                    costlist_weighted[-1] = responsibility_cost_list[i] * self.cost_weights["R"]

                costlist[-1] = responsibility_cost_list[i]
            else:
                costlist_weighted[-1] = 0.0
                costlist[-1] = 0.0

            total_cost = np.sum(costlist_weighted)

            if self.save_unweighted_costs:
                trajectory.set_costs(total_cost, costlist)
            else:
                trajectory.set_costs(total_cost, costlist_weighted)
