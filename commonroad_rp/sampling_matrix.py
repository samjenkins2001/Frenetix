import numpy as np
import itertools


def generate_sampling_matrix(*, t0_range, t1_range, s0_range, ss0_range, sss0_range, ss1_range, sss1_range, d0_range,
                             dd0_range, ddd0_range, d1_range, dd1_range, ddd1_range, debug_mode=True):
    """
    Generates a sampling matrix with all possible combinations of the given parameter ranges.
    Each row of the matrix is a different combination. Every parameter has to be passed by keyword argument,
    e.g. t0_range=[0, 1, 2], t1_range=[3, 4, 5], etc. to impede errors due to wrong order of arguments.

    Args:
    t0_range (np.array or int): Array of possible values for the starting time, or a single integer.
    t1_range (np.array or int): Array of possible values for the end time, or a single integer.
    s0_range (np.array or int): Array of possible values for the start longitudinal position, or a single integer.
    ss0_range (np.array or int): Array of possible values for the start longitudinal velocity, or a single integer.
    sss0_range (np.array or int): Array of possible values for the start longitudinal acceleration, or a single integer.
    s1_range (np.array or int): Array of possible values for the end longitudinal position, or a single integer.
    ss1_range (np.array or int): Array of possible values for the end longitudinal velocity, or a single integer.
    sss1_range (np.array or int): Array of possible values for the end longitudinal acceleration, or a single integer.
    d0_range (np.array or int): Array of possible values for the start lateral position, or a single integer.
    dd0_range (np.array or int): Array of possible values for the start lateral velocity, or a single integer.
    ddd0_range (np.array or int): Array of possible values for the start lateral acceleration, or a single integer.
    d1_range (np.array or int): Array of possible values for the end lateral position, or a single integer.
    dd1_range (np.array or int): Array of possible values for the end lateral velocity, or a single integer.
    ddd1_range (np.array or int): Array of possible values for the end lateral acceleration, or a single integer.
    debug_mode (boolean): If True, print the number of sampled trajectories. default: True

    Returns:
    np.array: 2D array (matrix) where each row is a different combination of parameters.
    """
    # Convert all input ranges to arrays, if they are not already
    ranges = [np.atleast_1d(x) for x in (
        t0_range, t1_range, s0_range, ss0_range, sss0_range, ss1_range, sss1_range, d0_range, dd0_range, ddd0_range,
        d1_range, dd1_range, ddd1_range)]

    # Use itertools.product to generate all combinations
    combinations = list(itertools.product(*ranges))

    if debug_mode >= 1:
        print('<ReactivePlanner>: %s trajectories sampled' % len(combinations))
    # Convert the list of combinations to a numpy array and return
    return np.array(combinations)
