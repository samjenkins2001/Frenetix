import os
import numpy as np
from behavior_planner.configuration_builder import ConfigurationBuilder
from commonroad.common.file_reader import CommonRoadFileReader
# from commonroad.planning.goal import GoalRegion
# commonroad-route-planner
from commonroad_route_planner.route_planner import RoutePlanner
import behavior_planner.utils.helper_functions as hf


class BehaviorPlanner(object):
    """
    Reactive planner class that plans trajectories in a sampling-based fashion
    """

    def __init__(self, proj_path, init_sc_path, init_ego_state, DT: float = 0.1):
        """
        Constructor of the behavior planner
        : param config: Configuration object holding all behavior planner-relevant configurations
        """

        # init
        self.DT = DT
        self.scenario = None
        self.planning_problem_set = None
        self.planning_problem = None
        self.route_planner = None
        self.all_routes = None
        self.ref_path = None
        self.desired_velocity_new = None

        config = ConfigurationBuilder.build_configuration(proj_path)
        self.priority_right = config.mission.priorityright
        self.overtaking = config.mode.overtaking
        self.set_scenario_and_planning_problem(init_sc_path)
        self.set_route_planner()
        self.select_ref_path_and_intermediate_goal()
        self.desired_velocity = init_ego_state.velocity

        """Velocity Planner Testing:""""""
        self.velocity_planner = VelocityPlanner(scenario=self.scenario, country=None)
        self.intermediate_goal_planner = IntermediateGoalPlanner(
            scenario=self.scenario, planning_problem=self.planning_problem)
        """"""Velocity Planner Testing:"""

    def set_scenario_and_planning_problem(self, scenario_path: str, idx_planning_problem: int = 0):
        self.scenario, self.planning_problem_set = CommonRoadFileReader(scenario_path).open()
        self.planning_problem = list(self.planning_problem_set.planning_problem_dict.values())[
            idx_planning_problem
        ]

    def set_route_planner(self, goal_region=None):
        self.route_planner = RoutePlanner(self.scenario, self.planning_problem, goal_region=goal_region)

    def select_ref_path_and_intermediate_goal(self):
        self.all_routes = self.route_planner.plan_routes()  # .retrieve_first_route().reference_path
        if self.all_routes.num_route_candidates == 1:
            self.ref_path = self.route_planner.plan_routes().retrieve_first_route().reference_path

        # traffic lights
        #  dists = []
        #  n_points = []
        #  for idx, traffic_signs in enumerate(self.scenario.lanelet_network.traffic_lights):
        #      dist, n_point = hf.dist_to_nearest_point(self.ref_path, traffic_signs.position)
        #      if dist < 10:
        #          dists.append(dist)
        #          n_points.append(n_point)
        #  for idx, traffic_dists in enumerate(dists):
        #      if traffic_dists < 10:

    def driving_mode(self, ego_state):
        obj_list_sorted = hf.sort_by_distance(ego_state, self.scenario.obstacles)
        remaining_path = hf.get_remaining_path(ego_state, self.ref_path)
        for idx, obj in enumerate(obj_list_sorted):
            dist, point_ = hf.dist_to_nearest_point(remaining_path, obj.initial_state.position)

            #
            #
            # if dist < 2.0:
            #     if hf.distance(ego_state.position, obj.initial_state.position) < 20:
            #        point_, index = hf.find_nearest_point_to_path(remaining_path, obj.initial_state.position)
            #         pos_llt = self.scenario.lanelet_network.find_lanelet_by_position([remaining_path[index-5]])
            #        #### TODO ###############
            #         new_goal = GoalRegion(pos_llt)
            #
            #         self.set_route_planner(obj.initial_state.position)
            #         self.ref_path = self.route_planner.plan_routes().retrieve_first_route().reference_path

    def set_desired_velocity(self, ego_state):
        try:
            # if the goal is not reached yet, try to reach it
            # get the center points of the possible goal positions
            goal_centers = []
            # get the goal lanelet ids if they are given directly in the planning problem
            if (
                    hasattr(self.planning_problem.goal, "lanelets_of_goal_position")
                    and self.planning_problem.goal.lanelets_of_goal_position is not None
            ):
                goal_lanelet_ids = self.planning_problem.goal.lanelets_of_goal_position[0]
                for lanelet_id in goal_lanelet_ids:
                    lanelet = self.scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
                    n_center_vertices = len(lanelet.center_vertices)
                    goal_centers.append(lanelet.center_vertices[int(n_center_vertices / 2.0)])
            elif hasattr(self.planning_problem.goal.state_list[0], "position"):
                # get lanelet id of the ending lanelet (of goal state), this depends on type of goal state
                if hasattr(self.planning_problem.goal.state_list[0].position, "center"):
                    goal_centers.append(self.planning_problem.goal.state_list[0].position.center)
            # if it is a survival scenario with no goal areas, no velocity can be proposed
            elif hasattr(self.planning_problem.goal.state_list[0], "time_step"):
                if ego_state.time_step > self.planning_problem.goal.state_list[0].time_step.end:
                    return 0.0
                else:
                    return self.planning_problem.initial_state.velocity
            else:
                return 0.0

            distances = []
            for goal_center in goal_centers:
                distances.append(hf.distance(goal_center, ego_state.position))

            # calculate the average distance to the goal positions
            avg_dist = np.mean(distances)

            _, max_remaining_time_steps = hf.calc_remaining_time_steps(
                planning_problem=self.planning_problem,
                ego_state_time=ego_state.time_step,
                t=0.0,
                dt=self.DT,
            )
            remaining_time = max_remaining_time_steps * self.DT

            # if there is time remaining, calculate the difference between the average desired velocity
            # and the velocity of the trajectory
            if remaining_time > 0.0:
                desired_velocity_new = avg_dist / remaining_time
            else:
                desired_velocity_new = 1

        except:
            print("Could not calculate desired velocity")
            desired_velocity_new = self.desired_velocity

        if np.abs(self.desired_velocity - desired_velocity_new) > 5 or np.abs(ego_state.velocity - desired_velocity_new) > 5:
            if np.abs(ego_state.velocity - desired_velocity_new) > 5:
                self.desired_velocity = ego_state.velocity + 1
            if desired_velocity_new > self.desired_velocity:
                self.desired_velocity_new = self.desired_velocity + 2
            else:
                desired_velocity_new = self.desired_velocity - 2

        self.desired_velocity = desired_velocity_new
