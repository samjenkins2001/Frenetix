__author__ = "Christian Pek"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["BMW Group CAR@TUM, interACT"]
__version__ = "0.5"
__maintainer__ = "Gerald Würsching"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Beta"

from abc import ABC, abstractmethod

import numpy as np

import commonroad_rp.trajectories

from Prediction.walenet.risk_assessment.collision_probability import (
    get_mahalanobis_dist_dict
)

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


class DefaultCostFunction(CostFunction):
    """
    Default cost function for comfort driving
    """

    def __init__(self, rp, predictions, desired_d):
        super(DefaultCostFunction, self).__init__()
        self.desired_speed = rp._desired_speed
        self.desired_d = desired_d
        self.predictions = predictions
        self.vehicle_params = rp.vehicle_params
        self.walenet_cost_factor = rp.walenet_cost_factor

    def evaluate(self, trajectory: commonroad_rp.trajectories.TrajectorySample):
        # acceleration costs
        costs = np.sum((5 * trajectory.cartesian.a) ** 2)
        # velocity costs
        costs += np.sum((5 * (trajectory.cartesian.v - self.desired_speed)) ** 2) + \
                 (50 * (trajectory.cartesian.v[-1] - self.desired_speed) ** 2) + \
                 (100 * (trajectory.cartesian.v[int(len(trajectory.cartesian.v)/2)] - self.desired_speed) ** 2)
        # distance costs
        costs += np.sum((0.25 * (self.desired_d - trajectory.curvilinear.d)) ** 2) + \
                 (20 * (self.desired_d - trajectory.curvilinear.d[-1])) ** 2
        # orientation costs
        costs += np.sum((0.25 * np.abs(trajectory.curvilinear.theta)) ** 2) + (
                5 * (np.abs(trajectory.curvilinear.theta[-1]))) ** 2
        # coll_prob_dict = get_collision_probability(
        #     traj=trajectory,
        #     predictions=predictions,
        #     vehicle_params=rp.vehicle_params,
        # )

        if self.predictions is not None:
            mahalanobis_costs = get_mahalanobis_dist_dict(
                traj=trajectory,
                predictions=self.predictions,
                vehicle_params=self.vehicle_params
            )
            costs += mahalanobis_costs * self.walenet_cost_factor
        return costs


class DefaultCostFunctionFailSafe(CostFunction):
    """
    Default cost function for fail-safe trajectory planning
    """

    def __init__(self):
        super(DefaultCostFunctionFailSafe, self).__init__()

    def evaluate(self, trajectory: commonroad_rp.trajectories.TrajectorySample):

        # acceleration costs
        costs = np.sum((1 * trajectory.cartesian.a) ** 2)
        # distance costs
        costs += np.sum((0.25 * trajectory.curvilinear.d) ** 2) + (20 * trajectory.curvilinear.d[-1]) ** 2
        # orientation costs
        costs += np.sum((0.25 * np.abs(trajectory.curvilinear.theta)) ** 2) + (
                5 * (np.abs(trajectory.curvilinear.theta[-1]))) ** 2

        return costs


class AdaptableCostFunction(CostFunction):
    """
    Default cost function for comfort driving
    """

    def __init__(self, desired_speed, desired_d, params):
        super(AdaptableCostFunction, self).__init__()
        self.desired_speed = desired_speed
        self.desired_d = desired_d
        self.params = params

    def evaluate(self, trajectory: commonroad_rp.trajectories.TrajectorySample):

        # acceleration costs
        costs = np.sum((self.params['acceleration'] * trajectory.cartesian.a) ** 2)
        # velocity costs
        costs += np.sum((self.params['velocity'][0] * (trajectory.cartesian.v - self.desired_speed)) ** 2) + \
                 (self.params['velocity'][1] * (trajectory.cartesian.v[-1] - self.desired_speed) ** 2) + \
                 (self.params['velocity'][2] * (trajectory.cartesian.v[int(len(trajectory.cartesian.v)/2)] - self.desired_speed) ** 2)
        # distance costs
        costs += np.sum((self.params['distance'][0] * (self.desired_d - trajectory.curvilinear.d)) ** 2) + \
                 (self.params['distance'][1] * (self.desired_d - trajectory.curvilinear.d[-1])) ** 2
        # orientation costs
        costs += np.sum((self.params['orientation'][0] * np.abs(trajectory.curvilinear.theta)) ** 2) + (
                self.params['orientation'][1] * (np.abs(trajectory.curvilinear.theta[-1]))) ** 2

        return costs
