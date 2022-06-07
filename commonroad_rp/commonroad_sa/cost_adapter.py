__author__ = "Alexander Hobmeier"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = []
__version__ = ""
__maintainer__ = "Alexander Hobmeier"
__email__ = "commonroad@lists.lrz.de"
__status__ = ""

# commonroad_rp imports
from commonroad_rp.cost_function import DefaultCostFunction, AdaptableCostFunction

from abc import ABC, abstractmethod

class CostAdapter(ABC):
    """
    Abstract base class for new cost functions
    """

    def __init__(self):
        pass

    @abstractmethod
    def adapt_cost_function(self, desired_speed: float, **kwargs):
        """
        Computes the costs of a given trajectory sample
        :param desired_speed: The desired speed
        :return: The cost fuction adapted to the current situation
        """
        pass


class DefaultCostAdapter(CostAdapter):
    """
    Use situational awareness to adapt cost function
    """

    def __init__(self):
        pass

    def adapt_cost_function(self, desired_speed, **kwargs):

        # TODO calculate cost params
        acceleration_cost = 5
        velocity_cost = [5, 50, 100]
        distance_cost = [0.25, 20]
        orientation_cost = [0.25, 5]

        params = {'acceleration': acceleration_cost, 'velocity': velocity_cost, 'distance': distance_cost, 'orientation': orientation_cost}
        return AdaptableCostFunction(desired_speed, 0, params)
