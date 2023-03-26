import os
# standard imports
import time
from copy import deepcopy

# third party
import numpy as np

# commonroad-io
from commonroad_rp.utility.collision_report import coll_report

# commonroad-io
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import InputState
from commonroad.scenario.trajectory import Trajectory
from commonroad.planning.planning_problem import PlanningProblem

# commonroad-route-planner
from commonroad_route_planner.route_planner import RoutePlanner
# reactive planner
from commonroad_rp.reactive_planner import ReactivePlanner
from commonroad_rp.utility.visualization import visualize_planner_at_timestep, make_gif
from commonroad_rp.cost_functions.cost_function import AdaptableCostFunction
from commonroad_rp.utility import helper_functions as hf
from commonroad_rp.configuration import Configuration

import commonroad_rp.prediction_helpers as ph
from behavior_planner.behavior_module import BehaviorModule

from commonroad.geometry.shape import Rectangle
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType


class Agent:
    """ Adapted from commonroad_rp/run_planner.py
    Represents one agent of the simulation, managing its planning problem,
    planner, and scenario (among others).
    """

    def __init__(self, agent_id: int, planning_problem: PlanningProblem,
                 scenario: Scenario, config: Configuration, log_path: str, mod_path: str):
        """Initialize an agent.
        :param agent_id: The agent ID, equal to the obstacle_id of the
                         DynamicObstacle it is represented by
        :param planning_problem: The planning problem of this agent
        :param config: The configuration
        :param log_path: Path for logging and visualization
        :param mod_path: working directory of the planner
                         (containing planner configuration)
        """

        # List of past states
        self.record_state_list = list()
        # List of input states for the planner
        self.record_input_list = list()
        # log of times required for planning steps
        self.planning_times = list()

        # Agent id, equals the id of the dummy obstacle
        self.id = agent_id

        # Vehicle shape
        self.shape = Rectangle(config.vehicle.length, config.vehicle.width)

        # Dummy obstacles for the agent:
        # Containing only the future trajectory, for collision checker and plotting
        self.ego_obstacle = None
        # Containing also the past trajectory, for synchronization and prediction
        self.full_ego_obstacle = None

        self.planning_problem = planning_problem
        self.config = config
        self.log_path = log_path
        self.mod_path = mod_path

        # Initialize Planner
        self.planner = ReactivePlanner(self.config, scenario, self.planning_problem,
                                       self.log_path, self.mod_path)

        # Local view on the scenario, with dummy obstacles
        # for all agents except the ego agent
        self.scenario = None
        # initialize scenario: Remove dynamic obstacle for ego vehicle
        self.set_scenario(scenario)

        # State before the next planning step
        # convert initial state from planning problem to reactive planner (Cartesian) state type
        self.x_0 = self.planner.process_initial_state_from_pp(x0_pp=deepcopy(self.planning_problem.initial_state))
        self.record_state_list.append(self.x_0)

        # Curvilinear state, used by the planner
        self.x_cl = None

        self.desired_velocity = self.x_0.velocity

        # add initial inputs to recorded input list
        self.record_input_list.append(InputState(
            acceleration=self.x_0.acceleration,
            time_step=self.x_0.time_step,
            steering_angle_speed=0.))

        # Do not use a behavior planner
        self.use_behavior_planner = False
        if not self.use_behavior_planner:
            route_planner = RoutePlanner(self.scenario, self.planning_problem)
            self.ref_path = route_planner.plan_routes().retrieve_first_route().reference_path
        else:
            # Load behavior planner
            self.behavior_module = BehaviorModule(proj_path=os.path.join(mod_path, "behavior_planner"),
                                                  init_sc_path=self.config.general.path_scenario,
                                                  init_ego_state=self.x_0,
                                                  vehicle_parameters=self.config.vehicle)  # testing
            self.ref_path = self.behavior_module.reference_path

        self.planner.set_reference_path(self.ref_path)

        # Set planning problem in the planner
        goal_area = hf.get_goal_area_shape_group(
            planning_problem=self.planning_problem, scenario=self.scenario
        )
        self.planner.set_goal_area(goal_area)
        self.planner.set_planning_problem(self.planning_problem)

        # set cost function
        self.cost_function = AdaptableCostFunction(rp=self.planner, configuration=config)
        self.planner.set_cost_function(self.cost_function)

        # initialize the prediction network if necessary
        self.predictor = ph.load_prediction(self.scenario, self.config.prediction.mode, config)

        self.current_timestep = self.x_0.time_step
        self.max_timestep = int(self.config.general.max_steps * planning_problem.goal.state_list[0].time_step.end)

    def set_scenario(self, scenario):
        """ Set the scenario.
        Required for updating the other agents after every planning step.
        :param scenario: The new scenario, will be copied by the agent.
        """
        self.scenario = deepcopy(scenario)
        if self.scenario.obstacle_by_id(self.id) is not None:
            self.scenario.remove_obstacle(self.scenario.obstacle_by_id(self.id))

        self.planner.set_scenario(self.scenario)

    def update_scenario(self, outdated_agents, dummy_obstacles):
        """Update the scenario to synchronize the agents.
        :param outdated_agents: Obstacle IDs of all dummy obstacles that need to be updated
        :param dummy_obstacles: New dummy obstacles
        """

        # Remove outdated obstacles
        for i in outdated_agents:
            # ego does not have a dummy of itself
            if i == self.id:
                continue
            self.scenario.remove_obstacle(self.scenario.obstacle_by_id(i))

        # Add all dummies except of the ego one
        self.scenario.add_objects(list(filter(lambda obs: not obs.obstacle_id == self.id, dummy_obstacles)))

        self.planner.set_scenario(self.scenario)

    def step_agent(self, global_predictions):
        """Execute one planning step.
        :param global_predictions: Dictionary of predictions for all obstacles and agents
        :returns status: 0, if successful
                         1, if completed
                         2, on error
                         3, on collision
        :returns full_ego_obstacle: the dummy obstacle at the new position of the agent,
                                    including both history and planned trajectory,
                                    or None if status > 0
        :returns ego_obstacle: the dummy obstacle at the new position of the agent, including
                               only the future trajectory, or None if status > 0
        """

        print(f"[Agent {self.id}] current time step: {self.current_timestep}")

        # check for completion of this agent
        if self.planner.goal_status or self.current_timestep >= self.max_timestep:
            self.finalize()
            return 1, None, None

        # Check for collisions in previous timestep
        if self.current_timestep > 0 and self.ego_obstacle is not None:
            crash = self.planner.check_collision(self.ego_obstacle)
            if crash:
                print(f"[Agent {self.id}] Collision Detected!")
                if self.config.debug.collision_report:
                    coll_report(self.record_state_list, self.planner, self.current_timestep,
                                self.scenario, self.ego_obstacle, self.planning_problem, self.log_path)
                self.finalize()
                return 3, None, None

        # Process new predictions
        predictions = dict()
        visible_obstacles, visible_area = ph.prediction_preprocessing(
            self.scenario, self.x_0, self.config, self.id
        )
        for obstacle_id in visible_obstacles:
            # Handle obstacles with higher initial timestep
            if obstacle_id in global_predictions.keys():
                predictions[obstacle_id] = global_predictions[obstacle_id]
        if self.config.prediction.cone_angle > 0 \
                and self.config.prediction.mode == "walenet" \
                or self.config.prediction.mode == "lanebased":
            predictions = ph.ignore_vehicles_in_cone_angle(predictions, self.x_0, self.config.vehicle.length,
                                                           self.config.prediction.cone_angle,
                                                           self.config.prediction.cone_safety_dist)

        # START TIMER
        comp_time_start = time.time()

        if not self.use_behavior_planner:
            # set desired velocity
            self.desired_velocity = hf.calculate_desired_velocity(self.scenario, self.planning_problem,
                                                                  self.x_0, self.config.planning.dt,
                                                                  self.desired_velocity)
            self.planner.set_desired_velocity(self.desired_velocity, self.x_0.velocity)
        else:
            """-----------------------------------------Testing:---------------------------------------------"""
            behavior_comp_time1 = time.time()
            self.behavior_module.execute(predictions=predictions, ego_state=self.x_0,
                                         time_step=self.current_timestep)

            # set desired behavior outputs
            self.planner.set_desired_velocity(self.behavior_module.desired_velocity, self.x_0.velocity)
            self.planner.set_reference_path(self.behavior_module.reference_path)
            behavior_comp_time2 = time.time()
            print("\n***Behavior Planning Time: \n", behavior_comp_time2 - behavior_comp_time1)

            """----------------------------------------Testing:---------------------------------------------"""
        # plan trajectory
        optimal = self.planner.plan(self.x_0, predictions, self.x_cl)  # returns the planned (i.e., optimal) trajectory
        comp_time_end = time.time()
        # END TIMER

        # if the planner fails to find an optimal trajectory -> terminate
        if not optimal:
            print(f"[Agent {self.id}] No Kinematic Feasible and Optimal Trajectory Available!")
            self.finalize()
            return 2, None, None

        # store planning times
        self.planning_times.append(comp_time_end - comp_time_start)
        print(f"[Agent {self.id}] ***Total Planning Time: {self.planning_times[-1]}")

        # correct orientation angle
        new_state_list = self.planner.shift_orientation(optimal[0], interval_start=self.x_0.orientation - np.pi,
                                                        interval_end=self.x_0.orientation + np.pi)

        # get next state from state list of planned trajectory
        new_state = new_state_list.state_list[1]
        new_state.time_step = self.current_timestep + 1

        # add input to recorded input list
        self.record_input_list.append(InputState(
            acceleration=new_state.acceleration,
            steering_angle_speed=(new_state.steering_angle - self.record_state_list[-1].steering_angle)
                                 / self.config.planning.dt,
            time_step=new_state.time_step
        ))
        # add new state to recorded state list
        self.record_state_list.append(new_state)

        # update init state and curvilinear state
        self.x_0 = deepcopy(self.record_state_list[-1])
        self.x_cl = (optimal[2][1], optimal[3][1])

        # Check if Goal is reached:
        self.planner.check_goal_reached()

        # commonroad obstacle for the ego vehicle, for collision checking and plotting
        # List of planned states, including the current state.
        future_state_list = optimal[0].state_list
        future_trajectory = Trajectory(future_state_list[0].time_step, future_state_list)
        self.ego_obstacle = self.planner.shift_and_convert_trajectory_to_object(future_trajectory, self.id)
        #future_prediction = TrajectoryPrediction(future_trajectory, self.shape)
        #self.ego_obstacle = DynamicObstacle(self.id, ObstacleType.CAR, self.shape,
        #                                    future_state_list[0], future_prediction)

        # commonroad obstacle for synchronization between agents
        # List of all past and future states.
        full_state_list = deepcopy(self.record_state_list)
        full_state_list.extend(future_state_list[2:])
        full_trajectory = Trajectory(full_state_list[0].time_step, full_state_list)
        self.full_ego_obstacle = self.planner.shift_and_convert_trajectory_to_object(full_trajectory, self.id)
        #full_prediction = TrajectoryPrediction(full_trajectory, self.shape)
        #self.full_ego_obstacle = DynamicObstacle(self.id, ObstacleType.CAR, self.shape,
        #                                         full_state_list[0], full_prediction)

        # plot own view on scenario
        if self.config.multiagent.show_individual_plots or self.config.multiagent.save_individual_plots:
            visualize_planner_at_timestep(scenario=self.scenario, planning_problem=self.planning_problem,
                                          ego=self.ego_obstacle,
                                          traj_set=self.planner.all_traj, ref_path=self.ref_path,
                                          timestep=self.current_timestep,
                                          config=self.config, predictions=predictions,
                                          plot_window=self.config.debug.plot_window_dyn,
                                          cluster=self.cost_function.cluster_prediction.cluster_assignments[-1]
                                          if self.cost_function.cluster_prediction is not None else None,
                                          log_path=self.log_path, visible_area=visible_area)

        self.current_timestep += 1
        return 0, self.full_ego_obstacle, self.ego_obstacle

    def finalize(self):
        # make gif
        if self.config.multiagent.save_individual_gifs:
            make_gif(self.config, self.scenario,
                     range(self.planning_problem.initial_state.time_step,
                           self.current_timestep),
                     self.log_path, duration=0.1)
