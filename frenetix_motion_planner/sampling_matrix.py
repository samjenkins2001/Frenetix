__author__ = "Georg Schmalhofer, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

import numpy as np
import itertools
import logging
from abc import ABC, abstractmethod

# get logger
msg_logger = logging.getLogger("Message_logger")


class SamplingHandler:
    def __init__(self, dt: float, spacing: list, num_trajectories: list, sampling_depth: int, t_min: float, horizon: float, delta_d_min: float,
                 delta_d_max: float, d_ego_pos: bool):
        self.dt = dt
        self.spacing = spacing
        self.num_trajectories = num_trajectories
        self.sampling_depth = sampling_depth
        self.d_ego_pos = d_ego_pos

        self.t_min = t_min
        self.horizon = horizon
        self.t_sampling = None

        self.delta_d_min = delta_d_min
        self.delta_d_max = delta_d_max
        self.d_sampling = None

        self.v_sampling = None
        self.s_sampling = None

        self.set_t_sampling()

        if not self.d_ego_pos:
            self.set_d_sampling()

    def update_static_params(self, t_min: float, horizon: float, delta_d_min: float, delta_d_max: float):
        assert t_min > 0, "t_min cant be <= 0"
        self.t_min = t_min
        self.horizon = horizon
        self.delta_d_min = delta_d_min
        self.delta_d_max = delta_d_max

        self.set_t_sampling()
        self.set_d_sampling()

    # def change_max_sampling_level(self, max_samp_lvl):
    #     self.max_sampling_number = max_samp_lvl

    def set_t_sampling(self):
        """
        Sets sample parameters of time horizon
        :param t_min: minimum of sampled time horizon
        :param horizon: sampled time horizon
        """
        self.t_sampling = TimeSampling(self.t_min, self.horizon, self.dt, self.sampling_depth, self.spacing)

    def set_d_sampling(self, lat_pos=None):
        """
        Sets sample parameters of lateral offset
        """
        if not self.d_ego_pos:
            self.d_sampling = LateralPositionSampling(self.delta_d_min, self.delta_d_max, self.sampling_depth, self.spacing)
        else:
            self.d_sampling = LateralPositionSampling(lat_pos + self.delta_d_min, lat_pos + self.delta_d_max, self.sampling_depth, self.spacing)

    def set_v_sampling(self, v_min, v_max):
        """
        Sets sample parameters of sampled velocity interval
        """
        self.v_sampling = VelocitySampling(v_min, v_max, self.sampling_depth, self.spacing)

    def set_s_sampling(self, delta_s_min, delta_s_max):
        """
        Sets sample parameters of lateral offset
        """
        self.s_sampling = LongitudinalPositionSampling(delta_s_min, delta_s_max, self.sampling_depth, self.spacing)



def generate_sampling_matrix(*, t0_range, t1_range, s0_range, ss0_range, sss0_range, ss1_range, sss1_range, d0_range,
                             dd0_range, ddd0_range, d1_range, dd1_range, ddd1_range):
    """
    Generates a sampling matrix with all possible combinations of the given parameter ranges.
    Each row of the matrix is a different combination. Every parameter has to be passed by keyword argument,
    e.g. t0_range=[0, 1, 2], t1_range=[3, 4, 5], etc. to impede errors due to wrong order of arguments.

    Args:
    00: t0_range (np.array or int): Array of possible values for the starting time, or a single integer.
    01: t1_range (np.array or int): Array of possible values for the end time, or a single integer.
    02: s0_range (np.array or int): Array of possible values for the start longitudinal position, or a single integer.
    03: ss0_range (np.array or int): Array of possible values for the start longitudinal velocity, or a single integer.
    04: sss0_range (np.array or int): Array of possible values for the start longitudinal acceleration, or a single integer.
    05: ss1_range (np.array or int): Array of possible values for the end longitudinal velocity, or a single integer.
    06: sss1_range (np.array or int): Array of possible values for the end longitudinal acceleration, or a single integer.
    07: d0_range (np.array or int): Array of possible values for the start lateral position, or a single integer.
    08: dd0_range (np.array or int): Array of possible values for the start lateral velocity, or a single integer.
    09: ddd0_range (np.array or int): Array of possible values for the start lateral acceleration, or a single integer.
    10: d1_range (np.array or int): Array of possible values for the end lateral position, or a single integer.
    11: dd1_range (np.array or int): Array of possible values for the end lateral velocity, or a single integer.
    12: ddd1_range (np.array or int): Array of possible values for the end lateral acceleration, or a single integer.
    13: debug_mode (boolean): If True, print the number of sampled trajectories. default: True

    Returns:
    np.array: 2D array (matrix) where each row is a different combination of parameters.
    """
    # Convert all input ranges to arrays, if they are not already
    ranges = [np.atleast_1d(x) for x in (
        t0_range, t1_range, s0_range, ss0_range, sss0_range, ss1_range, sss1_range, d0_range, dd0_range, ddd0_range,
        d1_range, dd1_range, ddd1_range)]

    # Use itertools.product to generate all combinations (if there are multiple values this is needed)
    #this is where the m x n is determined (all combinations of variables to create the conditions)
    combinations = list(itertools.product(*ranges))
    ############################
    # RULES
    # *ranges unpacks the tuple values, each value is used as a seperate argument for itertools.product
    # itertools.product generates the cartesian product of these arrays (all unique combinations of conditions)

    ############################

    msg_logger.debug('<ReactivePlanner>: %s trajectories sampled' % len(combinations))
    # Convert the list of combinations to a numpy array and return
    return np.array(combinations)


class Sampling(ABC):
    def __init__(self, minimum: float, maximum: float, sampling_depth: int, spacing: list):

        assert maximum >= minimum

        self.minimum = minimum
        self.maximum = maximum
        self.sampling_depth = sampling_depth
        self.spacing = spacing
        self._sampling_vec = list()
        self._initialization()

    @abstractmethod
    def _initialization(self):
        pass

    @abstractmethod
    def _regenerate_sampling_vec(self):
        pass

    def to_range(self, level: int, spacing: list, best_mult: int, min_val: float = None, max_val: float = None) -> set:
        """
        Obtain the sampling steps of a given sampling stage
        :param sampling_stage: The sampling stage to receive (>=0)
        :return: The set of sampling steps for the queried sampling stage
        """
        if self.spacing != spacing:
            self.spacing = spacing
            self._regenerate_sampling_vec()

        
        assert 0 < len(self.spacing), '<Sampling/to_range>: Provided sampling spacing is' \
                                                            ' incorrect! spacing = {}'.format(self.spacing)
        
        if level > 1:
            self.minimum = min_val
            self.maximum = max_val
            return set(np.linspace(self.minimum, self.maximum, len(self._sampling_vec[level - 1])))
        else:
            return self._sampling_vec[level - 1]

class VelocitySampling(Sampling):
    def __init__(self, minimum: float, maximum: float, sampling_depth: int, spacing: list):
        self.spacing = spacing
        self.sampling_depth = sampling_depth
        super(VelocitySampling, self).__init__(minimum, maximum)

    def _initialization(self):
        self._regenerate_sampling_vec()

    def _regenerate_sampling_vec(self):
        self._sampling_vec = []
        n = 3
        for i in range(self.sampling_depth):
            steps = int(round((self.maximum - self.minimum) / (self.spacing[i] * mult))) + 1
            self._sampling_vec.append(set(np.linspace(self.minimum, self.maximum, steps)))
            n = (n * 2) - 1


class LateralPositionSampling(Sampling):
    def __init__(self, minimum: float, maximum: float, sampling_depth: int, spacing: list):
        self.sampling_depth = sampling_depth
        self.spacing = spacing
        self.mult = 1.0
        super(LateralPositionSampling, self).__init__(minimum, maximum)

    def _initialization(self):
        self._regenerate_sampling_vec()

    def _regenerate_sampling_vec(self):
        self._sampling_vec = []
        n = 3
        for i in range(self.sampling_depth):
            steps = int(round((self.maximum - self.minimum) / self.spacing[i])) + 1
            self._sampling_vec.append(set(np.linspace(self.minimum, self.maximum, steps)))
            n = (n * 2) - 1
            
    def set_mult(self, mult):
        self.mult = mult
        self._regenerate_sampling_vec()


class LongitudinalPositionSampling(Sampling):
    def __init__(self, maximum: float,  minimum: float, sampling_depth: int, spacing: list):
        self.sampling_depth = sampling_depth
        self.spacing = spacing
        super(LongitudinalPositionSampling, self).__init__(maximum, minimum)

    def _initialization(self):
        self._regenerate_sampling_vec()

    def _regenerate_sampling_vec(self):
        self._sampling_vec = []
        n = 3
        for _ in range(self.sampling_depth):
            self._sampling_vec.append(set(np.linspace(self.minimum, self.maximum, n)))
            n = (n * 2) - 1


class TimeSampling(Sampling):
    def __init__(self, minimum: float, maximum: float, dT: float, sampling_depth: int, spacing: list):
        self.dT = dT
        self.sampling_depth = sampling_depth
        self.spacing = spacing
        super(TimeSampling, self).__init__(minimum, maximum)

    def _initialization(self):
        self._regenerate_sampling_vec()

    def _regenerate_sampling_vec(self):
        self._sampling_vec = []
        for i in range(self.sampling_depth):
            step_size = int((1 / (i + 1)) / self.dT)
            samp = set(np.round(np.arange(self.minimum, self.maximum + self.dT, (step_size * self.dT)), 2))
            samp.discard(elem for elem in list(samp) if elem > round(self.maximum + self.dT, 2))
            self._sampling_vec.append(samp)


# look into the kinematic check for rate of change of curvature/yaw when in a high risk scenario, in certain scenarios smoothness should not be required. ALSO is the kinematic check taylored to certain vehicles actuator capabilites?
# could look into adaptive cost functions for different scenarios, cost functions on a spectrum (changes with weather / road conditions / allow user configurations for comfort and efficiency)
# If there is no trajectory that avoids collision and adheres to road boundaries could we analyze the off road conditions and determine if shifting over the line is less risky than quickly stopping? (ie: large shoulder areas)


