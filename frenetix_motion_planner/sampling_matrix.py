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
    def __init__(self, user_input: list, dt: float, num_trajectories: list, spacing: list, sampling_depth: int, t_min: float, horizon: float, delta_d_min: float,
                 delta_d_max: float, d_ego_pos: bool):
        self.user_input = user_input
        self.dt = dt
        self.num_trajectories = num_trajectories
        self.spacing = spacing
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
        self.t_sampling = TimeSampling(self.t_min, self.horizon, self.dt, self.sampling_depth, self.num_trajectories, self.spacing, self.user_input)

    def set_d_sampling(self, lat_pos=None):
        """
        Sets sample parameters of lateral offset
        """
        if not self.d_ego_pos:
            self.d_sampling = LateralPositionSampling(self.delta_d_min, self.delta_d_max, self.sampling_depth, self.num_trajectories, self.spacing, self.user_input)
        else:
            self.d_sampling = LateralPositionSampling(lat_pos + self.delta_d_min, lat_pos + self.delta_d_max, self.sampling_depth, self.num_trajectories, self.spacing, self.user_input)

    def set_v_sampling(self, v_min, v_max):
        """
        Sets sample parameters of sampled velocity interval
        """
        self.v_sampling = VelocitySampling(v_min, v_max, self.sampling_depth, self.num_trajectories, self.spacing, self.user_input)

    def set_s_sampling(self, delta_s_min, delta_s_max):
        """
        Sets sample parameters of lateral offset
        """
        self.s_sampling = LongitudinalPositionSampling(delta_s_min, delta_s_max, self.sampling_depth, self.num_trajectories, self.spacing, self.user_input)



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

    def __init__(self, minimum: float, maximum: float, sampling_depth: int, num_trajectories: list, spacing: list, user_input: list):

        assert maximum >= minimum

        self.minimum = minimum
        self.maximum = maximum
        self.sampling_depth = sampling_depth
        self.num_trajectories = num_trajectories
        self.spacing = spacing
        self.user_input = user_input
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
            quotient = (goal + tolerance) // i
            product = i * quotient
            if goal - tolerance <= product <= goal + tolerance and quotient > 2 and i > 2:
                pair = [i, quotient]
                factor_pairs.append(pair)
                if pair[0] != pair[1]:
                    factor_pairs.append(pair[::-1])
            elif len(factor_pairs) == 0:
                tolerance += 10
                i = 0
                continue

        factor_pairs = [list(t) for t in set(tuple(p) for p in factor_pairs)]

        def sorting_key(pair):
            x0, x1 = pair
            mult = 2
            return abs((x1 * mult) - x0)
        
        factor_pairs.sort(key=sorting_key)
        return factor_pairs
    
        
    def get_spacing(self, stage):
        t1_len = [3, 4, 7, 10, 10]
        d1_values = []
        ss1_values = []
        if self.user_input == [False, True]:
            goal = int(self.num_trajectories[stage] / t1_len[stage])
            combinations = self.find_factor_pairs(goal, tolerance=0)
            for i in range(len(combinations)):
                d1_spacing = combinations[i][0]
                d1_values.append(d1_spacing)
                ss1_spacing = combinations[i][1]
                ss1_values.append(ss1_spacing)
            return d1_values, ss1_values
    
    def get_velocity_mult(self):
        mult = []
        for value in self.spacing:
            if value > 1:
                mult.append(4)
            elif value > 0.8:
                mult.append(6)
            elif value > 0.6:
                mult.append(8)
            elif value > 0.4:
                mult.append(10)
            elif value > 0.2:
                mult.append(18)
            else:
                mult.append(40)
        return mult
    
    
    def populate_spacing(self):
        d1_list = []
        ss1_list = []
        yaml_path = "configurations/frenetix_motion_planner/spacing.yaml"
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        if self.user_input == [False, True]:
            for i in range(self.sampling_depth):
                d1, ss1 = self.get_spacing(i)
                d1_list.append(d1[0])
                ss1_list.append(ss1[0])
            config['d1_values'] = d1_list
            config['ss1_values'] = ss1_list
        else:
            mult = self.get_velocity_mult()
            for i in range(len(self.spacing)):
                d1_list.append(self.spacing[i])
                ss1_list.append(self.spacing[i] * mult[i])
            config['d1_values'] = d1_list
            config['ss1_values'] = ss1_list
        
        with open(yaml_path, 'w') as file:
            yaml.safe_dump(config, file)
        
        self.d1_spacing = d1_list
        self.ss1_spacing = ss1_list

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


    def to_range(self, level: int, spacing: list, min_val: float = None, max_val: float = None, type: str = '', mode: str = 'normal', user_input: list = [False, True]) -> set:
        """
        Obtain the sampling steps of a given sampling stage
        :param sampling_stage: The sampling stage to receive (>=0)
        :return: The set of sampling steps for the queried sampling stage
        """

        if user_input != [False, True] and self.spacing != spacing:
            self.spacing = spacing
            self._regenerate_sampling_vec()

        
        assert 0 < max(len(self.num_trajectories), len(self.spacing)), '<Sampling/to_range>: Provided trajectories or spacing values are' \
                                                            ' incorrect! = {}'.format(max(self.num_trajectories, self.spacing))
        
        if level > 1:
            self.minimum = min_val
            self.maximum = max_val
            # self._regenerate_sampling_vec('normal')
            return set(np.linspace(self.minimum, self.maximum, len(self._sampling_vec[level - 1])))
        else:
            if self.update_spacing_ss1() and type == 'ss1' and user_input == [False, True]:
                VelocitySampling._regenerate_sampling_vec(self, mode)
                return self._sampling_vec[level - 1]
            elif self.update_spacing_d1() and type == 'd1' and user_input == [False, True]:
                LateralPositionSampling._regenerate_sampling_vec(self, mode)
                return self._sampling_vec[level - 1]
            else:
                return self._sampling_vec[level - 1]

class VelocitySampling(Sampling):
    def __init__(self, minimum: float, maximum: float, sampling_depth: int, num_trajectories: list, spacing: list, user_input: list):
        self.num_trajectories = num_trajectories
        self.spacing = spacing
        super(VelocitySampling, self).__init__(minimum, maximum, sampling_depth, num_trajectories, spacing, user_input)

    def _initialization(self):
        self._regenerate_sampling_vec(mode='normal')

    def _regenerate_sampling_vec(self, mode):
        t1_len = [3, 4, 7, 10, 10]
        if mode == 'emergency':
            self.minimum = 0
            self.maximum = self.maximum * 2
        self._sampling_vec = []
        for i in range(self.sampling_depth):
            if self.user_input == [True, False]:
                steps = int(round((self.maximum - self.minimum) / (self.ss1_spacing[i]))) + 1
            elif self.user_input == [True, True]:
                vec = self.get_shared_sampling_vec()[i]
                steps = int(round(self.num_trajectories[i] / (len(vec) * t1_len[i]))) 
            else:
                steps = self.ss1_spacing[i] - 1
            self._sampling_vec.append(set(np.linspace(self.minimum, self.maximum, steps)))

class LateralPositionSampling(Sampling):
    def __init__(self, minimum: float, maximum: float, sampling_depth: int, num_trajectories: list, spacing: list, user_input: list):
        self.num_trajectories = num_trajectories
        self.spacing = spacing
        super(LateralPositionSampling, self).__init__(minimum, maximum, sampling_depth, num_trajectories, spacing, user_input)

    def _initialization(self):
        self._regenerate_sampling_vec(mode='normal')
        if self.user_input == [True, True]:
            self.set_shared_sampling_vec(self._sampling_vec)

    def _regenerate_sampling_vec(self, mode):
        if mode == 'emergency':
            self.minimum = 0
            self.maximum = self.maximum * 2
        self._sampling_vec = []
        for i in range(self.sampling_depth):
            if self.user_input != [False, True]:
                steps = int(round((self.maximum - self.minimum) / (self.d1_spacing[i]))) + 1
            else:
                steps = self.d1_spacing[i] - 1
            self._sampling_vec.append(set(np.linspace(self.minimum, self.maximum, steps)))


class LongitudinalPositionSampling(Sampling):
    def __init__(self, maximum: float,  minimum: float, sampling_depth: int, num_trajectories: list, spacing: list, user_input: list):
        super(LongitudinalPositionSampling, self).__init__(maximum, minimum, sampling_depth, num_trajectories, spacing, user_input)

    def _initialization(self):
        self._regenerate_sampling_vec()

    def _regenerate_sampling_vec(self, mode):
        self._sampling_vec = []
        n = 3
        for _ in range(self.sampling_depth):
            self._sampling_vec.append(set(np.linspace(self.minimum, self.maximum, n)))
            n = (n * 2) - 1


class TimeSampling(Sampling):
    def __init__(self, minimum: float, maximum: float, dT: float, sampling_depth: int, num_trajectories: list, spacing: list, user_input: list):
        self.dT = dT
        super(TimeSampling, self).__init__(minimum, maximum, sampling_depth, num_trajectories, spacing, user_input)

    def _initialization(self):
        self._regenerate_sampling_vec(mode='normal')
        # self.update_spacing()
        # self.set_shared_sampling_vec(self._sampling_vec)

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
# Explore other search algorithms for planning


