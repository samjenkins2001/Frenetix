import os
import traceback
from copy import deepcopy
import numpy as np

from commonroad.common.solution import CommonRoadSolutionWriter

from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.obstacle import ObstacleRole
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory

from commonroad_dc import pycrcc
from commonroad_dc.boundary import boundary
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object
from commonroad_dc.feasibility.solution_checker import valid_solution, CollisionException, GoalNotReachedException, \
    MissingSolutionException
from commonroad_rp.cost_functions.cost_function import AdaptableCostFunction
from commonroad_rp.reactive_planner import ReactivePlanner, ReactivePlannerState
from commonroad_rp.utility import helper_functions as hf
from commonroad_rp.utility.evaluation import create_full_solution_trajectory, create_planning_problem_solution, \
    reconstruct_inputs, reconstruct_states, check_acceleration, plot_states, plot_inputs
from commonroad_rp.utility.visualization import plot_final_trajectory

from commonroad_route_planner.route_planner import RoutePlanner

from behavior_planner.behavior_module import BehaviorModule

from multiagent.multiagent_helpers import collision_vis
from multiagent.planner_interface import PlannerInterface

from risk_assessment.harm_estimation import harm_model
from risk_assessment.helpers.collision_helper_function import angle_range
from risk_assessment.utils.logistic_regression_symmetrical import get_protected_inj_prob_log_reg_ignore_angle


class FrenetPlannerInterface(PlannerInterface):

    def __init__(self, config, scenario, planning_problem, log_path, mod_path):
        self.config = config
        self.scenario = scenario
        self.predictions = None
        self.planning_problem = planning_problem
        self.log_path = log_path
        self.mod_path = mod_path

        self.planner = ReactivePlanner(config, scenario, planning_problem,
                                       log_path, mod_path)

        self.x_0 = ReactivePlannerState.create_from_initial_state(deepcopy(planning_problem.initial_state), config.vehicle.wheelbase, config.vehicle.wb_rear_axle)
        self.x_cl = None

        self.planner.set_x_0(self.x_0)

        self.desired_velocity = self.x_0.velocity

        # Set reference path
        self.use_behavior_planner = False
        if not self.use_behavior_planner:
            route_planner = RoutePlanner(scenario, planning_problem)
            self.ref_path = route_planner.plan_routes().retrieve_first_route().reference_path
        else:
            # Load behavior planner
            self.behavior_module = BehaviorModule(proj_path=os.path.join(mod_path, "behavior_planner"),
                                                  init_sc_path=config.general.path_scenario,
                                                  init_ego_state=self.x_0, dt=scenario.dt,
                                                  vehicle_parameters=config.vehicle)  # testing
            self.ref_path = self.behavior_module.reference_path

        self.planner.set_reference_path(self.ref_path)

        # Set planning problem
        goal_area = hf.get_goal_area_shape_group(
            planning_problem=planning_problem, scenario=scenario
        )
        self.planner.set_goal_area(goal_area)
        self.planner.set_planning_problem(planning_problem)

        # set cost function
        self.cost_function = AdaptableCostFunction(rp=self.planner, configuration=config)
        self.planner.set_cost_function(self.cost_function)

    def is_completed(self):
        self.planner.check_goal_reached()
        return self.planner.goal_status

    def coll_report(self, ego_vehicle_list, timestep, collision_report_path: str):
        """Replaces coll_report from collision_report.py

        Collect and present detailed information about a collision.

        :param ego_vehicle_list: List of ego obstacles for at least the last two time steps.
        :param timestep: Time step at which the collision occurred.
        :param collision_report_path: The path to write the report to.
        """

        # check if the current state is collision-free
        vel_list = []
        # get ego position and orientation
        try:
            ego_pos = ego_vehicle_list[-1].state_at_time(timestep).position

        except AttributeError:
            print("None-type error")
            traceback.print_exc()

        (
            _,
            road_boundary,
        ) = boundary.create_road_boundary_obstacle(
            scenario=self.scenario,
            method="aligned_triangulation",
            axis=2,
        )

        if timestep == 0:
            ego_vel = ego_vehicle_list[-1].initial_state.velocity
            ego_yaw = ego_vehicle_list[-1].initial_state.orientation

            vel_list.append(ego_vel)
        else:
            ego_pos_last = ego_vehicle_list[-2].state_at_time(timestep).position

            delta_ego_pos = ego_pos - ego_pos_last

            ego_vel = np.linalg.norm(delta_ego_pos) / self.scenario.dt

            vel_list.append(ego_vel)

            ego_yaw = np.arctan2(delta_ego_pos[1], delta_ego_pos[0])

        current_state_collision_object = hf.create_tvobstacle(
            traj_list=[
                [
                    ego_pos[0],
                    ego_pos[1],
                    ego_yaw,
                ]
            ],
            box_length=self.planner.vehicle_params.length / 2,
            box_width=self.planner.vehicle_params.width / 2,
            start_time_step=timestep,
        )

        # Add road boundary to collision checker
        self.planner._cc.add_collision_object(road_boundary)

        if not self.planner._cc.collide(current_state_collision_object):
            return

        # get the colliding obstacle
        obs_id = None
        for obs in self.scenario.obstacles:
            co = create_collision_object(obs)
            if current_state_collision_object.collide(co):
                if obs.obstacle_id != ego_vehicle_list[-1].obstacle_id:
                    if obs_id is None:
                        obs_id = obs.obstacle_id
                    else:
                        print("More than one collision detected")
                        return

        # Collision with boundary
        if obs_id is None:
            ego_harm = get_protected_inj_prob_log_reg_ignore_angle(
                velocity=ego_vel, coeff=self.planner.params_harm
            )
            total_harm = ego_harm

            print("Collision with road boundary. (Harm: {:.2f})".format(ego_harm))
            return

        # get information of colliding obstacle
        obs_pos = (
            self.scenario.obstacle_by_id(obstacle_id=obs_id)
            .occupancy_at_time(time_step=timestep)
            .shape.center
        )
        obs_pos_last = (
            self.scenario.obstacle_by_id(obstacle_id=obs_id)
            .occupancy_at_time(time_step=timestep - 1)
            .shape.center
        )
        obs_size = self.scenario.obstacle_by_id(obstacle_id=obs_id).obstacle_shape.length \
                       * self.scenario.obstacle_by_id(obstacle_id=obs_id).obstacle_shape.width

        # filter initial collisions
        if timestep < 1:
            print("Collision at initial state")
            return
        if (
                self.scenario.obstacle_by_id(obstacle_id=obs_id).obstacle_role
                == ObstacleRole.ENVIRONMENT
        ):
            obs_vel = 0
            obs_yaw = 0
        else:
            pos_delta = obs_pos - obs_pos_last

            obs_vel = np.linalg.norm(pos_delta) / self.scenario.dt
            if (
                    self.scenario.obstacle_by_id(obstacle_id=obs_id).obstacle_role
                    == ObstacleRole.DYNAMIC
            ):
                obs_yaw = np.arctan2(pos_delta[1], pos_delta[0])
            else:
                obs_yaw = self.scenario.obstacle_by_id(
                    obstacle_id=obs_id
                ).initial_state.orientation

        # calculate crash angle
        pdof = angle_range(obs_yaw - ego_yaw + np.pi)
        rel_angle = np.arctan2(
            obs_pos_last[1] - ego_pos_last[1], obs_pos_last[0] - ego_pos_last[0]
        )
        ego_angle = angle_range(rel_angle - ego_yaw)
        obs_angle = angle_range(np.pi + rel_angle - obs_yaw)

        # calculate harm
        ego_harm, obs_harm, ego_obj, obs_obj = harm_model(scenario=self.scenario, ego_vehicle_sc=ego_vehicle_list[-1],
                                                          vehicle_params=self.planner.vehicle_params, ego_velocity=ego_vel,
                                                          ego_yaw=ego_yaw, obstacle_id=obs_id, obstacle_size=obs_size,
                                                          obstacle_velocity=obs_vel, obstacle_yaw=obs_yaw, pdof=pdof,
                                                          ego_angle=ego_angle, obs_angle=obs_angle,
                                                          modes=self.planner.params_risk, coeffs=self.planner.params_harm)

        # if collision report should be shown
        collision_vis(
            scenario=self.scenario,
            ego_vehicle=ego_vehicle_list[-1],
            destination=collision_report_path,
            ego_harm=ego_harm,
            ego_type=ego_obj.type,
            ego_v=ego_vel,
            ego_mass=ego_obj.mass,
            obs_harm=obs_harm,
            obs_type=obs_obj.type,
            obs_v=obs_vel,
            obs_mass=obs_obj.mass,
            pdof=pdof,
            ego_angle=ego_angle,
            obs_angle=obs_angle,
            time_step=timestep,
            modes=self.planner.params_risk,
            planning_problem=self.planning_problem,
            global_path=None,
            driven_traj=None,
        )

    def check_collision(self, ego_vehicle_list, timestep):
        # Adapted from ReactivePlanner.check_collision
        # to allow using ego obstacles containing the complete trajectory.

        ego_vehicle = ego_vehicle_list[-1]

        ego = pycrcc.TimeVariantCollisionObject((timestep + 1))
        ego.append_obstacle(
            pycrcc.RectOBB(0.5 * self.planner.vehicle_params.length, 0.5 * self.planner.vehicle_params.width,
                           ego_vehicle.state_at_time(timestep).orientation,
                           ego_vehicle.state_at_time(timestep).position[0],
                           ego_vehicle.state_at_time(timestep).position[1]))

        if not self.planner.collision_checker.collide(ego):
            return False
        else:
            try:
                goal_position = []

                if self.planner.goal_checker.goal.state_list[0].has_value("position"):
                    for x in self.planner.reference_path:
                        if self.planner.goal_checker.goal.state_list[0].position.contains_point(x):
                            goal_position.append(x)
                    s_goal_1, d_goal_1 = self.planner._co.convert_to_curvilinear_coords(
                        goal_position[0][0],
                        goal_position[0][1])
                    s_goal_2, d_goal_2 = self.planner._co.convert_to_curvilinear_coords(
                        goal_position[-1][0],
                        goal_position[-1][1])
                    s_goal = min(s_goal_1, s_goal_2)
                    s_start, d_start = self.planner._co.convert_to_curvilinear_coords(
                        self.planner.planning_problem.initial_state.position[0],
                        self.planner.planning_problem.initial_state.position[1])
                    s_current, d_current = self.planner._co.convert_to_curvilinear_coords(
                        ego_vehicle.state_at_time(timestep).position[0],
                        ego_vehicle.state_at_time(timestep).position[1])
                    progress = ((s_current - s_start) / (s_goal - s_start))
                elif "time_step" in self.planner.goal_checker.goal.state_list[0].attributes:
                    progress = (timestep - 1 / self.planner.goal_checker.goal.state_list[0].time_step.end)
                else:
                    print('Could not calculate progress')
                    progress = None
            except:
                progress = None
                print('Could not calculate progress')

            collision_obj = self.planner.collision_checker.find_all_colliding_objects(ego)[0]
            if isinstance(collision_obj, pycrcc.TimeVariantCollisionObject):
                obj = collision_obj.obstacle_at_time(timestep)
                center = obj.center()
                last_center = collision_obj.obstacle_at_time(timestep - 1).center()
                r_x = obj.r_x()
                r_y = obj.r_y()
                orientation = obj.orientation()
                self.planner.logger.log_collision(True, self.planner.vehicle_params.length,
                                                  self.planner.vehicle_params.width,
                                                  progress, center,
                                                  last_center, r_x, r_y, orientation)
            else:
                self.planner.logger.log_collision(False, self.planner.vehicle_params.length,
                                                  self.planner.vehicle_params.width,
                                                  progress)

            if self.config.debug.collision_report:
                self.coll_report(ego_vehicle_list, timestep, self.log_path)

            return True

    def get_all_traj(self):
        return self.planner.all_traj

    def get_ref_path(self):
        return self.planner.reference_path

    def update_planner(self, scenario: Scenario, predictions: dict):
        self.scenario = scenario
        self.predictions = predictions

        self.planner.set_scenario(scenario)
        self.planner.set_predictions(predictions)

        self.planner.set_x_0(self.x_0)
        self.planner.set_x_cl(self.x_cl)

    def plan(self):
        if not self.use_behavior_planner:
            # set desired velocity
            self.desired_velocity = hf.calculate_desired_velocity(self.scenario, self.planning_problem,
                                                                  self.x_0, self.config.planning.dt,
                                                                  self.desired_velocity)
            self.planner.set_desired_velocity(self.desired_velocity, self.x_0.velocity)
        else:
            """-----------------------------------------Testing:---------------------------------------------"""
            self.behavior_module.execute(predictions=self.predictions, ego_state=self.x_0,
                                         time_step=self.x_0.time_step)

            # set desired behavior outputs
            self.planner.set_desired_velocity(self.behavior_module.desired_velocity, self.x_0.velocity)
            self.planner.set_reference_path(self.behavior_module.reference_path)

            """----------------------------------------Testing:---------------------------------------------"""

        # plan trajectory
        optimal = self.planner.plan()

        if not optimal:
            # Could not plan feasible trajectory
            return 1, None

        # correct orientation angle
        new_trajectory = self.planner.shift_orientation(optimal[0], interval_start=self.x_0.orientation - np.pi,
                                                        interval_end=self.x_0.orientation + np.pi)

        # get next state from state list of planned trajectory
        new_state = new_trajectory.state_list[1]
        new_state.time_step = self.x_0.time_step + 1

        # update init state and curvilinear state
        self.x_0 = deepcopy(new_state)
        self.x_cl = (optimal[2][1], optimal[3][1])

        # Shift the state list to the center of the vehicle
        shifted_state_list = []
        for x in new_trajectory.state_list:
            shifted_state_list.append(
                x.translate_rotate(np.array([self.config.vehicle.wb_rear_axle * np.cos(x.orientation),
                                             self.config.vehicle.wb_rear_axle * np.sin(x.orientation)]),
                                   0.0)
            )

        shifted_trajectory = Trajectory(shifted_state_list[0].time_step,
                                        shifted_state_list)
        return 0, shifted_trajectory

    def evaluate(self, id, recorded_state_list, recorded_input_list):

        # create full solution trajectory
        initial_timestep = self.planning_problem.initial_state.time_step
        ego_solution_trajectory = Trajectory(initial_time_step=initial_timestep,
                                             state_list=recorded_state_list[initial_timestep:])

        # plot full ego vehicle trajectory
        plot_final_trajectory(self.scenario, self.planning_problem, ego_solution_trajectory.state_list,
                              self.config, self.log_path)

        # create CR solution
        solution = create_planning_problem_solution(self.config, ego_solution_trajectory,
                                                    self.scenario, self.planning_problem)

        # check feasibility
        # reconstruct inputs (state transition optimizations)
        feasible, reconstructed_inputs = reconstruct_inputs(self.config, solution.planning_problem_solutions[0])
        try:
            # reconstruct states from inputs
            reconstructed_states = reconstruct_states(self.config, ego_solution_trajectory.state_list,
                                                      reconstructed_inputs)
            # check acceleration correctness
            check_acceleration(self.config, ego_solution_trajectory.state_list, plot=True)

            # remove first element from input list
            recorded_input_list.pop(0)

            # evaluate
            plot_states(self.config, ego_solution_trajectory.state_list, self.log_path, reconstructed_states,
                        plot_bounds=False)
            # CR validity check
            print(f"[Agent {id}] Feasibility Check Result: ")
            if valid_solution(self.scenario, PlanningProblemSet([self.planning_problem]), solution)[0]:
                print(f"[Agent {id}] Valid")
        except CollisionException:
            print(f"[Agent {id}] Infeasible: Collision")
        except GoalNotReachedException:
            print(f"[Agent {id}] Infeasible: Goal not reached")
        except MissingSolutionException:
            print(f"[Agent {id}] Infeasible: Missing solution")
        except:
            traceback.print_exc()
            print(f"[Agent {id}] Could not reconstruct states")

        plot_inputs(self.config, recorded_input_list, self.log_path, reconstructed_inputs, plot_bounds=True)

        # Write Solution to XML File for later evaluation
        solutionwriter = CommonRoadSolutionWriter(solution)
        solutionwriter.write_to_file(self.log_path, "solution.xml", True)
