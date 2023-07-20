__author__ = "Alexander Hobmeier, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"


from abc import ABC, abstractmethod
from enum import Enum
from typing import List

from omegaconf import OmegaConf
from commonroad_rp.trajectories import TrajectorySample
from commonroad_rp.cost_functions.partial_cost_functions import *


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

        self.vehicle_params = rp.vehicle_params
        self.cost_weights = OmegaConf.to_object(configuration.cost.cost_weights)
        self.save_unweighted_costs = configuration.debug.save_unweighted_costs
        self.cost_weights_names = None

        self.delete_irrelevant_costs()

    def delete_irrelevant_costs(self):
        for category in list(self.cost_weights.keys()):
            if self.cost_weights[category] == 0:
                del self.cost_weights[category]
        names = list(self.cost_weights.keys())
        names.sort()
        self.cost_weights_names = names

    def update_state(self, scenario, rp, predictions, reachset):
        self.scenario = scenario
        self.rp = rp
        self.reachset = reachset
        self.desired_speed = rp._desired_speed
        self.predictions = predictions

    # calculate all costs for all trajcetories
    def evaluate(self, trajectories: List[TrajectorySample]):
        self.calc_cost(trajectories)

    # calculate all costs and weigh them
    def calc_cost(self, trajectories: List[TrajectorySample]):

        for i, trajectory in enumerate(trajectories):
            costlist = np.zeros(len(self.cost_weights))
            costlist_weighted = np.zeros(len(self.cost_weights))

            for num, function in enumerate(self.cost_weights_names):
                function_iteration = globals()[function + "_costs"]
                costlist[num] = function_iteration(trajectory=trajectory, planner=self.rp,
                                                      scenario=self.scenario, desired_speed=self.desired_speed)
                costlist_weighted[num] = self.cost_weights[function] * costlist[num]

            total_cost = np.sum(costlist_weighted)

            trajectory.set_costs(total_cost, costlist, costlist_weighted, self.cost_weights_names)

