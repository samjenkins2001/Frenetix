__author__ = "Gerald Würsching"
__copyright__ = "TUM Cyber-Physical Systems Group"
__version__ = "0.5"
__maintainer__ = "Gerald Würsching"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Beta"


# standard imports
import time
from copy import deepcopy

# third party
import numpy as np

# commonroad-io
from commonroad.scenario.trajectory import State

# commonroad-io
from commonroad.scenario.state import InputState

# commonroad-route-planner
from commonroad_route_planner.route_planner import RoutePlanner
from commonroad_rp.utility import helper_functions as hf

# reactive planner
from commonroad_rp.reactive_planner import ReactivePlanner
from commonroad_rp.utility.visualization import visualize_planner_at_timestep, plot_final_trajectory, make_gif
from commonroad_rp.utility.evaluation import create_planning_problem_solution, reconstruct_inputs, plot_states, \
    plot_inputs, reconstruct_states, create_full_solution_trajectory, check_acceleration
from commonroad_rp.cost_functions.cost_function import AdaptableCostFunction
from commonroad_rp.utility.helper_functions import (
    get_goal_area_shape_group,
)
from commonroad_rp.utility.general import load_scenario_and_planning_problem

from commonroad_rp.prediction_helpers import main_prediction, load_walenet, prediction_preprocessing
from Prediction.walenet.risk_assessment.collision_probability import ignore_vehicles_in_cone_angle

from commonroad_prediction.prediction_module import PredictionModule


def run_planner(config, log_path, mod_path):

    # *************************************
    # Set Configurations
    # *************************************

    DT = config.planning.dt            # planning time step

    # *************************************
    # Open CommonRoad scenario
    # *************************************

    scenario, planning_problem, planning_problem_set = load_scenario_and_planning_problem(config)

    # *************************************
    # Init and Goal State
    # *************************************
    # initial state configuration
    problem_init_state = planning_problem.initial_state

    if not hasattr(problem_init_state, 'acceleration'):
        problem_init_state.acceleration = 0.
    x_0 = deepcopy(problem_init_state)

    # goal state configuration
    goal = planning_problem.goal
    if hasattr(planning_problem.goal.state_list[0], 'velocity'):
        desired_velocity = (planning_problem.goal.state_list[0].velocity.start + planning_problem.goal.state_list[0].velocity.end) / 2
    else:
        desired_velocity = x_0.velocity + 5


    # *************************************
    # Initialize Planner
    # *************************************

    # initialize reactive planner
    planner = ReactivePlanner(config, scenario, planning_problem, log_path, mod_path)
    # set sampling parameters
    planner.set_d_sampling_parameters(config.sampling.d_min, config.sampling.d_max)
    planner.set_t_sampling_parameters(config.sampling.t_min, config.planning.dt, config.planning.planning_horizon)
    # set collision checker
    planner.set_collision_checker(planner.scenario)
    # initialize route planner and set reference path
    route_planner = RoutePlanner(scenario, planning_problem)
    ref_path = route_planner.plan_routes().retrieve_first_route().reference_path

    # ref_path = extrapolate_ref_path(ref_path)
    planner.set_reference_path(ref_path)
    goal_area = get_goal_area_shape_group(
       planning_problem=planning_problem, scenario=scenario
    )
    planner.set_goal_area(goal_area)
    planner.set_planning_problem(planning_problem)
    # set cost function
    cost_function = AdaptableCostFunction(rp=planner, configuration=config)
    planner.set_cost_function(cost_function)

    # **************************
    # Run Planning
    # **************************
    # initialize lists to store states and inputs
    record_state_list = list()
    record_input_list = list()
    x_cl = None
    current_count = 0
    planning_times = list()
    ego_vehicle = None

    # convert initial state from planning problem to reactive planner (Cartesian) state type
    x_0 = planner.process_initial_state_from_pp(x0_pp=x_0)
    record_state_list.append(x_0)

    # add initial inputs to recorded input list
    record_input_list.append(InputState(
        acceleration=x_0.acceleration,
        time_step=x_0.time_step,
        steering_angle_speed=0.))

    # initialize the prediction network if necessary
    if config.prediction.walenet:
        predictor = load_walenet(scenario=scenario)
    elif config.prediction.lanebased:
        pred_horizon_in_seconds = config.prediction.pred_horizon_in_s
        pred_horizon_in_timesteps = int(pred_horizon_in_seconds / DT)
        predictor_lanelet = PredictionModule(scenario, timesteps=pred_horizon_in_timesteps, dt=DT)

    new_state = None

    # Run planner
    while not goal.is_reached(x_0) and current_count < config.general.max_steps:
        current_count = len(record_state_list) - 1

        # new planning cycle -> plan a new optimal trajectory

        # START TIMER
        comp_time_start = time.time()
        # set desired velocity
        desired_velocity = hf.calculate_desired_velocity(scenario, planning_problem, x_0, DT, desired_velocity)
        planner.set_desired_velocity(desired_velocity, x_0.velocity)
        if current_count > 1:
            ego_state = new_state
        else:
            ego_state = x_0

        # get visible objects if the prediction is used
        if config.prediction.walenet or config.prediction.calc_visible_area:
            visible_obstacles, visible_area = prediction_preprocessing(scenario, ego_state, config)

        if config.prediction.walenet:
            predictions = main_prediction(predictor, scenario, visible_obstacles, ego_state, DT,
                                          [float(config.planning.planning_horizon)])
        elif config.prediction.lanebased:
            predictions = predictor_lanelet.main_prediction(ego_state, config.prediction.sensor_radius,
                                          [float(config.planning.planning_horizon)])
        else:
            predictions = None

        # ignore Predictions in Cone Angle
        if config.prediction.cone_angle > 0 and (config.prediction.lanebased or config.prediction.walenet):
            predictions = ignore_vehicles_in_cone_angle(predictions, ego_state, config.vehicle.length,
                                                        config.prediction.cone_angle,
                                                        config.prediction.cone_safety_dist)

        # plan trajectory
        optimal, tmn, tmn_ott = planner.plan(x_0, predictions, x_cl)  # returns the planned (i.e., optimal) trajectory
        comp_time_end = time.time()
        # END TIMER

        # store planning times
        planning_times.append(comp_time_end - comp_time_start)
        print(f"***Total Planning Time: {planning_times[-1]}")

        # if the planner fails to find an optimal trajectory -> terminate
        if not optimal or tmn:
            if not optimal:
                print("not optimal")
            if tmn and not tmn_ott:
                print("Scenario succeeded")
            elif tmn and not tmn_ott:
                print("Scenario succeeded, but outside the time horizon")
            break

        # correct orientation angle
        new_state_list = planner.shift_orientation(optimal[0], interval_start=x_0.orientation-np.pi,
                                                   interval_end=x_0.orientation+np.pi)


        # get next state from state list of planned trajectory
        new_state = new_state_list.state_list[1]
        new_state.time_step = current_count + 1

        # add input to recorded input list
        record_input_list.append(InputState(
            acceleration=new_state.acceleration,
            steering_angle_speed=(new_state.steering_angle - record_state_list[-1].steering_angle) / DT,
            time_step=new_state.time_step
        ))
        # add new state to recorded state list
        record_state_list.append(new_state)

        # update init state and curvilinear state
        x_0 = deepcopy(record_state_list[-1])
        x_cl = (optimal[2][1], optimal[3][1])

        # create CommonRoad Obstacle for the ego Vehicle
        if config.debug.show_plots or config.debug.save_plots:
            ego_vehicle = planner.shift_and_convert_trajectory_to_object(optimal[0])

        print(f"current time step: {current_count}")
        # draw scenario + planning solution
        if config.debug.show_plots or config.debug.save_plots:
            visualize_planner_at_timestep(scenario=scenario, planning_problem=planning_problem, ego=ego_vehicle,
                                          traj_set=planner.all_traj, ref_path=ref_path, timestep=current_count,
                                          config=config, predictions=predictions, plot_window=config.debug.plot_window_dyn,
                                          log_path=log_path, visible_area=visible_area)
        if x_0.time_step > 1:
            if planner.check_collision():
                print("Collision Detected!")
                break

    # plot  final ego vehicle trajectory
    plot_final_trajectory(scenario, planning_problem, record_state_list, config, log_path)

    # make gif
    if config.debug.gif:
        make_gif(config, scenario, range(0, current_count), log_path, duration=0.25)

    # remove first element
    record_input_list.pop(0)

    # **************************
    # Evaluate results
    # **************************

    if config.debug.evaluation:
        from commonroad.common.solution import CommonRoadSolutionWriter
        from commonroad_dc.feasibility.solution_checker import valid_solution

        # create full solution trajectory
        ego_solution_trajectory = create_full_solution_trajectory(config, record_state_list)

        # plot full ego vehicle trajectory
        plot_final_trajectory(scenario, planning_problem, ego_solution_trajectory.state_list, config, log_path)

        # create CR solution
        solution = create_planning_problem_solution(config, ego_solution_trajectory, scenario, planning_problem)

        # check feasibility
        # reconstruct inputs (state transition optimizations)
        feasible, reconstructed_inputs = reconstruct_inputs(config, solution.planning_problem_solutions[0])
        try:
            # reconstruct states from inputs
            reconstructed_states = reconstruct_states(config, ego_solution_trajectory.state_list, reconstructed_inputs)
            # check acceleration correctness
            check_acceleration(config, ego_solution_trajectory.state_list, plot=True)

            # remove first element from input list
            record_input_list.pop(0)

            # evaluate
            plot_states(config, ego_solution_trajectory.state_list, log_path, reconstructed_states, plot_bounds=False)
            # CR validity check
            print("Feasibility Check Result: ")
            print(valid_solution(scenario, planning_problem_set, solution))
        except:
            print("Could not reconstruct states")

        plot_inputs(config, record_input_list, log_path, reconstructed_inputs, plot_bounds=True)

        # Write Solution to XML File for later evaluation
        solutionwrtiter = CommonRoadSolutionWriter(solution)
        solutionwrtiter.write_to_file(log_path, "solution.xml", True)

