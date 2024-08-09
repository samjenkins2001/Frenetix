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
import yaml

# get logger
msg_logger = logging.getLogger("Message_logger")


class SamplingHandler:
    def __init__(self, dt: float, num_trajectories: list, sampling_depth: int, t_min: float, horizon: float, delta_d_min: float,
                 delta_d_max: float, d_ego_pos: bool):
        self.dt = dt
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

    def set_t_sampling(self):
        """
        Sets sample parameters of time horizon
        :param t_min: minimum of sampled time horizon
        :param horizon: sampled time horizon
        """
        self.t_sampling = TimeSampling(self.t_min, self.horizon, self.dt, self.sampling_depth, self.num_trajectories)

    def set_d_sampling(self, lat_pos=None):
        """
        Sets sample parameters of lateral offset
        """
        if not self.d_ego_pos:
            self.d_sampling = LateralPositionSampling(self.delta_d_min, self.delta_d_max, self.sampling_depth, self.num_trajectories)
        else:
            self.d_sampling = LateralPositionSampling(lat_pos + self.delta_d_min, lat_pos + self.delta_d_max, self.sampling_depth, self.num_trajectories)

    def set_v_sampling(self, v_min, v_max):
        """
        Sets sample parameters of sampled velocity interval
        """
        self.v_sampling = VelocitySampling(v_min, v_max, self.sampling_depth, self.num_trajectories)

    def set_s_sampling(self, delta_s_min, delta_s_max):
        """
        Sets sample parameters of lateral offset
        """
        self.s_sampling = LongitudinalPositionSampling(delta_s_min, delta_s_max, self.sampling_depth, self.num_trajectories)



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

    _shared_sampling_vec = []

    def __init__(self, minimum: float, maximum: float, sampling_depth: int, num_trajectories: list):

        assert maximum >= minimum

        self.minimum = minimum
        self.maximum = maximum
        self.sampling_depth = sampling_depth
        self.num_trajectories = num_trajectories
        self._sampling_vec = list()
        self.populate_spacing()
        self._initialization()
    
    def get_shared_sampling_vec(self):
        return Sampling._shared_sampling_vec
    
    def set_shared_sampling_vec(self, vec):
        Sampling._shared_sampling_vec = vec

    @abstractmethod
    def _initialization(self):
        pass

    @abstractmethod
    def _regenerate_sampling_vec(self, mode):
        pass

    def get_sampling_vec(self):
        return self._sampling_vec
    
    def find_factor_pairs(self, goal, tolerance):
        factor_pairs = []

        for i in range(1, goal + tolerance + 1):
            if i > goal + tolerance:
                break
            if goal - tolerance <= i * (goal // i) <= goal + tolerance:
                factor_pairs.append([i, goal // i])
        
        factor_pairs.sort(key=lambda x: abs(x[0] - x[1]))
        return factor_pairs
    
        
    def get_spacing(self, stage):
        t1_len = [3, 4, 7, 10, 10]
        goal = int(self.num_trajectories[stage] / t1_len[stage])
        combinations = self.find_factor_pairs(goal, tolerance=20)
        d1_values = []
        ss1_values = []
        for i in range(len(combinations)):
            d1_spacing = combinations[i][0]
            d1_values.append(d1_spacing)
            ss1_spacing = combinations[i][1]
            ss1_values.append(ss1_spacing)
        return d1_values, ss1_values
    
    
    def populate_spacing(self):
        d1_list = []
        ss1_list = []
        yaml_path = "configurations/frenetix_motion_planner/spacing.yaml"
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        for i in range(self.sampling_depth):
            d1, ss1 = self.get_spacing(i)
            d1_list.append(d1[0])
            ss1_list.append(ss1[0])
        config['d1_values'] = d1_list
        config['ss1_values'] = ss1_list
        with open(yaml_path, 'w') as file:
            yaml.safe_dump(config, file)
        
        self.d1_spacing = config['d1_values']
        self.ss1_spacing = config['ss1_values']

    
    def update_spacing_ss1(self):
        with open("configurations/frenetix_motion_planner/spacing.yaml", 'r') as file:
            data = yaml.safe_load(file)
            self.cur_ss1 = data['ss1_values']
            if self.ss1_spacing != self.cur_ss1:
                self.ss1_spacing = self.cur_ss1
                return True
            else:
                return False
            
    def update_spacing_d1(self):
        with open("configurations/frenetix_motion_planner/spacing.yaml", 'r') as file:
            data = yaml.safe_load(file)
            self.cur_d1 = data['d1_values']
            if self.d1_spacing != self.cur_d1:
                self.d1_spacing = self.cur_d1
                return True
            else:
                return False


    def to_range(self, level: int, min_val: float = None, max_val: float = None, type: str = '', mode: str = 'normal') -> set:
        """
        Obtain the sampling steps of a given sampling stage
        :param sampling_stage: The sampling stage to receive (>=0)
        :return: The set of sampling steps for the queried sampling stage
        """

        
        assert 0 < len(self.num_trajectories), '<Sampling/to_range>: Provided trajectories are' \
                                                            ' incorrect! trajectories = {}'.format(self.num_trajectories)
        
        if level > 1:
            self.minimum = min_val
            self.maximum = max_val
            return set(np.linspace(self.minimum, self.maximum, len(self._sampling_vec[level - 1])))
        else:
            if self.update_spacing_ss1() and type == 'ss1':
                VelocitySampling._regenerate_sampling_vec(self, mode)
                return self._sampling_vec[level - 1]
            elif self.update_spacing_d1() and type == 'd1':
                LateralPositionSampling._regenerate_sampling_vec(self, mode)
                return self._sampling_vec[level - 1]
            else:
                return self._sampling_vec[level - 1]

class VelocitySampling(Sampling):
    def __init__(self, minimum: float, maximum: float, sampling_depth: int, num_trajectories: list):
        self.num_trajectories = num_trajectories
        super(VelocitySampling, self).__init__(minimum, maximum, sampling_depth, num_trajectories)

    def _initialization(self):
        self._regenerate_sampling_vec(mode='normal')

    def _regenerate_sampling_vec(self, mode):
        if mode == 'emergency':
            self.minimum = 0
            self.maximum = self.maximum * 2
        self._sampling_vec = []
        for i in range(self.sampling_depth):
            self._sampling_vec.append(set(np.linspace(self.minimum, self.maximum, (self.ss1_spacing[i] - 1))))

class LateralPositionSampling(Sampling):
    def __init__(self, minimum: float, maximum: float, sampling_depth: int, num_trajectories: list):
        self.num_trajectories = num_trajectories
        super(LateralPositionSampling, self).__init__(minimum, maximum, sampling_depth, num_trajectories)

    def _initialization(self):
        self._regenerate_sampling_vec(mode='normal')

    def _regenerate_sampling_vec(self, mode):
        if mode == 'emergency':
            self.minimum = 0
            self.maximum = self.maximum * 2
        self._sampling_vec = []
        for i in range(self.sampling_depth):
            self._sampling_vec.append(set(np.linspace(self.minimum, self.maximum, (self.d1_spacing[i] - 1))))


class LongitudinalPositionSampling(Sampling):
    def __init__(self, maximum: float,  minimum: float, sampling_depth: int, num_trajectories: list):
        super(LongitudinalPositionSampling, self).__init__(maximum, minimum, sampling_depth, num_trajectories)

    def _initialization(self):
        self._regenerate_sampling_vec()

    def _regenerate_sampling_vec(self, mode):
        self._sampling_vec = []
        n = 3
        for _ in range(self.sampling_depth):
            self._sampling_vec.append(set(np.linspace(self.minimum, self.maximum, n)))
            n = (n * 2) - 1


class TimeSampling(Sampling):
    def __init__(self, minimum: float, maximum: float, dT: float, sampling_depth: int, num_trajectories: list):
        self.dT = dT
        super(TimeSampling, self).__init__(minimum, maximum, sampling_depth, num_trajectories)

    def _initialization(self):
        self._regenerate_sampling_vec(mode='normal')
        # self.update_spacing()
        self.set_shared_sampling_vec(self._sampling_vec)

    def _regenerate_sampling_vec(self, mode):
        self._sampling_vec = []
        for i in range(self.sampling_depth):
            step_size = int((1 / (i + 1)) / self.dT)
            samp = set(np.round(np.arange(self.minimum, self.maximum + self.dT, (step_size * self.dT)), 2))
            samp.discard(elem for elem in list(samp) if elem > round(self.maximum + self.dT, 2))
            self._sampling_vec.append(samp)


        



# look into the kinematic check for rate of change of curvature/yaw when in a high risk scenario, in certain scenarios smoothness should not be required. ALSO is the kinematic check taylored to certain vehicles actuator capabilites?
# could look into adaptive cost functions for different scenarios, cost functions on a spectrum (changes with weather / road conditions / allow user configurations for comfort and efficiency)
# If there is no trajectory that avoids collision and adheres to road boundaries could we analyze the off road conditions and determine if shifting over the line is less risky than quickly stopping? (ie: large shoulder areas)


