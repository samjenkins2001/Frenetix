import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
# from commonroad_helper_functions.interpolation import interpolate_scenario
import commonroad_rp.utility.helper_functions as hf


def load_scenario_and_planning_problem(config, idx_planning_problem: int = 0):
    scenario, planning_problem_set = CommonRoadFileReader(config.general.path_scenario).open()
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[
        idx_planning_problem
    ]

    # if config.planning.dt != scenario.dt:
    #     scenario, planning_problem_set = interpolate_scenario(scenario=scenario, dt_new=config.planning.dt,
    #                                                           planning_problem_set=planning_problem_set)
    #
    # scenario = check_if_obstacle_on_same_position(scenario, planning_problem.initial_state.position)

    return scenario, planning_problem, planning_problem_set


def check_if_obstacle_on_same_position(scenario, init_state):
    init_pos = np.array((init_state[0], init_state[1]))
    for i in scenario.dynamic_obstacles:
        if i.prediction.initial_time_step < 2:
            obs_pos = np.array((i.initial_state.position[0], i.initial_state.position[1]))
            if hf.distance(init_pos, obs_pos) < 2:
                scenario.remove_obstacle(i)
    return scenario
