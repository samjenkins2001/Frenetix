__author__ = "Rainer Trauth, Gerald Würsching"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

import os
# standard imports
import time
from copy import deepcopy

# third party
import numpy as np

# commonroad-io
from cr_scenario_handler.utils.collision_report import coll_report

# commonroad-io
from commonroad.scenario.state import InputState

# commonroad-route-planner
from commonroad_route_planner.route_planner import RoutePlanner
# reactive planner
from commonroad_rp.reactive_planner import ReactivePlanner
from commonroad_rp.state import ReactivePlannerState
from commonroad_rp.utility.visualization import visualize_planner_at_timestep, plot_final_trajectory, make_gif
from cr_scenario_handler.utils.evaluation import create_planning_problem_solution, reconstruct_inputs, plot_states, \
    plot_inputs, reconstruct_states, create_full_solution_trajectory, check_acceleration
from commonroad_rp.cost_functions.cost_function import AdaptableCostFunction
from commonroad_rp.utility import helper_functions as hf
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.geometry.shape import Rectangle

from cr_scenario_handler.utils.general import load_scenario_and_planning_problem

import commonroad_rp.prediction_helpers as ph
from behavior_planner.behavior_module import BehaviorModule

from commonroad_rp.occlusion_planning.occlusion_module import OcclusionModule


def run_planner(config, log_path, mod_path):

    DT = config.planning.dt  # planning time step

    # *************************************
    # Open CommonRoad scenario
    # *************************************

    scenario, planning_problem, planning_problem_set = load_scenario_and_planning_problem(config)

    # *************************************
    # Init and Goal State
    # *************************************
    problem_init_state = planning_problem.initial_state

    if not hasattr(problem_init_state, 'acceleration'):
        problem_init_state.acceleration = 0.
    x_0 = deepcopy(problem_init_state)

    goal_area = hf.get_goal_area_shape_group(
       planning_problem=planning_problem, scenario=scenario
    )

    # *************************************
    # Initialize Reactive Planner
    # *************************************
    planner = ReactivePlanner(config, scenario, planning_problem, log_path, mod_path)

    # **************************
    # Run Variables
    # **************************
    shape = Rectangle(planner.vehicle_params.length, planner.vehicle_params.width)
    ego_vehicle = [DynamicObstacle(42, ObstacleType.CAR, shape, x_0, None)]
    planner.current_ego_vehicle = ego_vehicle
    x_cl = None
    current_count = 0
    planning_times = list()

    behavior = None
    behavior_modul = None
    predictions = None
    visible_area = None
    occlusion_map = None
    occlusion_module = None

    # **************************
    # Convert Initial State
    # **************************
    x_0 = ReactivePlannerState.create_from_initial_state(x_0, config.vehicle.wheelbase, config.vehicle.wb_rear_axle)
    planner.record_state_and_input(x_0)

    # *************************************
    # Load Behavior Planner
    # *************************************
    if hasattr(planning_problem.goal.state_list[0], 'velocity'):
        desired_velocity = (planning_problem.goal.state_list[0].velocity.start + planning_problem.goal.state_list[
            0].velocity.end) / 2
    else:
        desired_velocity = x_0.velocity + 5

    if not config.behavior.use_behavior_planner:
        route_planner = RoutePlanner(scenario, planning_problem)
        reference_path = route_planner.plan_routes().retrieve_first_route().reference_path
    else:
        behavior_modul = BehaviorModule(proj_path=os.path.join(mod_path, "behavior_planner"),
                                        init_sc_path=config.general.name_scenario,
                                        init_ego_state=x_0,
                                        dt=DT,
                                        vehicle_parameters=config.vehicle)  # testing
        reference_path = behavior_modul.reference_path

    # **************************
    # Load Prediction
    # **************************
    predictor = ph.load_prediction(scenario, config.prediction.mode, config)

    # **************************
    # Initialize Occlusion Module
    # **************************
    if config.occlusion.use_occlusion_module:
        occlusion_module = OcclusionModule(scenario, config, reference_path, log_path, planner)

    # **************************
    # Set External Planner Setups
    # **************************
    planner.update_externals(goal_area=goal_area, planning_problem=planning_problem, occlusion_module=occlusion_module,
                             reference_path=reference_path)
    cost_function = AdaptableCostFunction(rp=planner, configuration=config)
    planner.update_externals( cost_function=cost_function)

    # **************************
    # Run Planner Cycle
    # **************************
    max_time_steps_scenario = int(config.general.max_steps*planning_problem.goal.state_list[0].time_step.end)
    while not planner.goal_status and current_count < max_time_steps_scenario:

        current_count = len(planner.record_state_list) - 1

        # **************************
        # Cycle Prediction
        # **************************
        if config.prediction.mode:
            predictions, visible_area = ph.step_prediction(scenario, predictor, config, x_0, occlusion_module)

        # **************************
        # Cycle Behavior Planner
        # **************************
        if not config.behavior.use_behavior_planner:
            # set desired velocity
            desired_velocity = hf.calculate_desired_velocity(scenario, planning_problem, x_0, DT, desired_velocity)
        else:
            behavior_comp_time1 = time.time()
            behavior = behavior_modul.execute(predictions=predictions, ego_state=x_0, time_step=current_count)
            behavior_comp_time2 = time.time()
            # set desired behavior outputs
            desired_velocity = behavior_modul.desired_velocity
            reference_path = behavior_modul.reference_path
            print("\n***Behavior Planning Time: \n", behavior_comp_time2 - behavior_comp_time1)

        # **************************
        # Cycle Occlusion Module
        # **************************
        if config.occlusion.use_occlusion_module and (current_count == 0 or current_count % 1 == 0):
            occlusion_module.step(predictions=predictions)

        # **************************
        # Set Planner Subscriptions
        # **************************
        planner.update_externals(x_0=x_0, x_cl=x_cl, reference_path=reference_path,
                                 desired_velocity=desired_velocity, predictions=predictions, behavior=behavior)

        # **************************
        # Execute Planner
        # **************************
        comp_time_start = time.time()
        optimal = planner.plan()  # returns the planned (i.e., optimal) trajectory
        comp_time_end = time.time()

        # if the planner fails to find an optimal trajectory -> terminate
        if not optimal:
            print("No Kinematic Feasible and Optimal Trajectory Available!")
            break

        # store planning times
        planning_times.append(comp_time_end - comp_time_start)
        print(f"***Total Planning Time: {planning_times[-1]}")

        # record state and input
        planner.record_state_and_input(optimal[0].state_list[1])

        # update init state and curvilinear state
        x_0 = deepcopy(planner.record_state_list[-1])
        x_cl = (optimal[2][1], optimal[3][1])

        print(f"current time step: {current_count}")

        # **************************
        # Visualize Scenario
        # **************************
        if config.debug.show_plots or config.debug.save_plots:
            visualize_planner_at_timestep(scenario=scenario, planning_problem=planning_problem, ego=planner.current_ego_vehicle[-1],
                                          traj_set=planner.all_traj, optimal_traj=optimal[0], ref_path=reference_path, timestep=current_count,
                                          config=config, predictions=predictions,
                                          plot_window=config.debug.plot_window_dyn,
                                          cluster=cost_function.cluster_prediction.cluster_assignments[-1]
                                                  if cost_function.cluster_prediction is not None else None,
                                          log_path=log_path, visible_area=visible_area, occlusion_map=occlusion_map)

        # **************************
        # Check Collision
        # **************************
        crash = planner.check_collision(planner.current_ego_vehicle[-1])
        if crash:
            print("Collision Detected!")
            if config.debug.collision_report and current_count > 0:
                coll_report(planner.current_ego_vehicle, planner, scenario, planning_problem,
                            log_path)
            break

        # **************************
        # Check Goal Status
        # **************************
        planner.check_goal_reached()

    # ******************************************************************************
    # End of Cycle
    # ******************************************************************************

    print(planner.goal_message)
    if planner.full_goal_status:
        print("\n", planner.full_goal_status)
    if not planner.goal_status and current_count >= max_time_steps_scenario:
        print("Scenario Aborted! Maximum Time Step Reached!")

    # plot  final ego vehicle trajectory
    plot_final_trajectory(scenario, planning_problem, planner.record_state_list, config, log_path)

    # make gif
    if config.debug.gif:
        make_gif(config, scenario, range(0, current_count), log_path, duration=0.25)

    # **************************
    # Evaluate results
    # **************************
    if config.debug.evaluation:
        from commonroad.common.solution import CommonRoadSolutionWriter
        from commonroad_dc.feasibility.solution_checker import valid_solution

        # create full solution trajectory
        ego_solution_trajectory = create_full_solution_trajectory(config, planner.record_state_list)

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

            # evaluate
            plot_states(config, ego_solution_trajectory.state_list, log_path, reconstructed_states, plot_bounds=False)
            # CR validity check
            print("Feasibility Check Result: ")
            print(valid_solution(scenario, planning_problem_set, solution))
        except:
            print("Could not reconstruct states")

        plot_inputs(config, planner.record_input_list, log_path, reconstructed_inputs, plot_bounds=True)

        # Write Solution to XML File for later evaluation
        solutionwriter = CommonRoadSolutionWriter(solution)
        solutionwriter.write_to_file(log_path, "solution.xml", True)

