from commonroad.geometry.shape import Polygon, Circle
# from commonroad_crime.utility import solver, general
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
import pandas as pd
import numpy as np
from cr_scenario_handler.utils.visualization import visualize_scenario_and_pp
import cr_scenario_handler.utils.helper_functions as hf
import math
from risk_assessment.risk_costs import calc_risk

# def evaluation_wrapper(args):
#     agent_id, time_step, used_measures, config_measures = args
#     return evaluate_timestep(agent_id, time_step, used_measures, config_measures)
class Measures:
    # adapted from CriMe
    def __init__(self, agent_id, scenario, t_start, t_end, reference_paths=None, msg_logger = None):
        self.agent_id = agent_id
        self.scenario = scenario
        self.dt = scenario.dt
        self.t_start = t_start
        self.t_end = t_end
        self.ego = scenario.obstacle_by_id(agent_id)
        self.other_obstacles = self._set_other_obstacles()

        self.radius = 100
        self.cosy = self._update_clcs(reference_paths, self.radius)
        self.msg_logger = msg_logger





    def _set_other_obstacles(self):
        other_obstacles = [obs for obs in self.scenario.obstacles
                           if obs.obstacle_id is not self.agent_id]
        return other_obstacles

    def _get_obstacles_in_proximity(self, time_step,  #scenario, ego_id: int, ego_state, time_step: int, radius: float,
                            vehicles_in_cone_angle=True):#, config=None):
        """
        Get all the obstacles that can be found in a given radius.

        Args:
            scenario (Scenario): Considered Scenario.
            ego_id (int): ID of the ego vehicle.
            ego_state (State): State of the ego vehicle.
            time_step (int) time step
            radius (float): Considered radius.

        Returns:
            [int]: List with the IDs of the obstacles that can be found in the ball with the given radius centering at the ego vehicles position.
        """

        obstacles_within_radius = []
        ego_state = self.ego.state_at_time(time_step)
        for obs in self.other_obstacles:
            # do not consider the ego vehicle
            if obs.obstacle_id != self.ego.obstacle_id:
                obs_state = obs.state_at_time(time_step)
                # if the obstacle is not in the lanelet network at the given time, its occupancy is None
                if obs is not None:
                    # calculate the distance between the two obstacles
                    dist = hf.distance(
                        pos1=ego_state.position,
                        pos2=obs_state.position,
                    )
                    # add obstacles that are close enough
                    if dist < self.radius:
                        obstacles_within_radius.append(obs)
        if vehicles_in_cone_angle: # and config:
            obstacles_within_radius = self._vehicles_in_cone_angle(time_step, obs, cone_angle=45)
        return obstacles_within_radius

    def _vehicles_in_cone_angle(self, time_step, obstacles, # scenario, ego_pose, veh_length,
                                      cone_angle=45): # cone_safety_dist):
        """Ignore vehicles behind ego for prediction if inside specific cone.

        Cone is spaned from center of rear-axle (cog - length / 2.0)

        cone_angle = Totel Angle of Cone. 0.5 per side (right, left)

        return bool: True if vehicle is ignored, i.e. inside cone
        """
        ego_state = self.ego.state_at_time(time_step)

        cone_angle = cone_angle / 180 * np.pi

        obs_list_behind = list()
        obs_list_front = list()
        for obs in obstacles:
            obs_state = obs.state_at_time(time_step)
            # loc_obj_pos = hf.rotate_glob_loc(
            #     obs_state.position - ego_state.position, obs_state.orientation, matrix=False)

            dist_sd = headway, _ = self._sd_distance(obs_state.position, ego_state.position, ego_state.orientation)

            obs_angle = hf.pi_range(math.atan2(dist_sd[1], dist_sd[0]) - np.pi)


            if abs(obs_angle) > cone_angle / 2.0 and dist_sd[0] < 0:
                # cone behind vehicle
                obs_list_behind.append(obs)

            elif abs(obs_angle) > cone_angle / 2.0 and dist_sd[0] > 0:
                # cone in front of vehicle
                obs_list_front.append(obs)

            # TODO add side-cones?
            # TODO plot cone angel to check curv. coordinates
            visualize_scenario_and_pp(self.scenario, cosy=self.cosy)

        return {"back": obs_list_behind, "front": obs_list_front}

        ###
        # ego_pose = np.array(
        #     [ego_pose.initial_state.position[0], ego_pose.initial_state.position[1],
        #      ego_pose.initial_state.orientation])
        # cone_angle = cone_angle / 180 * np.pi
        # ignore_pred_list = list()

        # for i in obstacles:
        #     ignore_object = True
        #     obj_pose = scenario.obstacle_by_id(i).occupancy_at_time(time_step).shape.center
        #     obj_orientation = scenario.obstacle_by_id(i).occupancy_at_time(time_step).shape.orientation

            # loc_obj_pos = hf.rotate_glob_loc(
            #     obj_pose[:2] - ego_pose[:2], obj_orientation, matrix=False
            # )
        #     loc_obj_pos[0] += veh_length / 2.0
        #
        #     if loc_obj_pos[0] > -cone_safety_dist:
        #         ignore_object = False
        #
        #     obj_angle = hf.pi_range(math.atan2(loc_obj_pos[1], loc_obj_pos[0]) - np.pi)
        #
        #     if abs(obj_angle) > cone_angle / 2.0:
        #         ignore_object = False
        #     if ignore_object:
        #         ignore_pred_list.append(i)
        #
        # if len(ignore_pred_list) > 0:
        #     # for obj in range(len(ignore_pred_list)):
        #     for obj in ignore_pred_list:
        #         obstacles.remove(obj)
        #
        # return obstacles

    def _sd_distance(self, other_position, ego_position, orientation):
        dist_sd = hf.rotate_glob_loc(
            (self.cosy.convert_to_curvilinear_coords(other_position[0], other_position[1]) -
             self.cosy.convert_to_curvilinear_coords(ego_position[0], ego_position[1])),
            orientation, matrix=False)
        return dist_sd


    def _update_clcs(self, reference_path, radius):
        """
        Updates the curvilinear coordinate system in the configuration setting using the reference path from the lanelet
        where the ego vehicle is currently located to the end of the lanelet.
        """
        # default setting of ego vehicle's curvilinear coordinate system
        # if not reference_path:
        #     ego_initial_lanelet_id = list(
        #         self.ego.prediction.center_lanelet_assignment[0]
        #     )[0]
        #     reference_path = general.generate_reference_path(ego_initial_lanelet_id, self.scenario.lanelet_network)
        #TODO
        cosy = CurvilinearCoordinateSystem(reference_path, radius)
        return cosy

    @staticmethod
    def _rear_end_position(obs, t):
        # other rear-end position
        if obs.state_at_time(t):
            if isinstance(obs.obstacle_shape, Polygon):
                other_position = obs.state_at_time(t).position
            elif isinstance(obs.obstacle_shape, Circle):
                other_position = obs.state_at_time(t).position - obs.obstacle_shape.radius
            else:
                other_position = obs.state_at_time(t).position - obs.obstacle_shape.length / 2
            return other_position
        return None

    # TODO how diff lanelets? Crime only uses same lanelet id -> does not work at intersection
    def hw(self):
        # see commonroad_crime.measure.distance.hw for details
        results = pd.Series(None, index=list(range(self.t_start, self.t_end + 1)))
        for t in range(self.t_start, self.t_end + 1):
            if self.msg_logger is not None:
                self.msg_logger.debug(f"evaluating hw of {self.agent_id} at timestep {t}")
            ego_position = self.ego.state_at_time(t).position + self.ego.obstacle_shape.length / 2
            ego_orientation = self.ego.state_at_time(t).orientation
            veh_results = []
            for other_obs in self.other_obstacles:
                other_position = self._rear_end_position(other_obs, t)
                if other_position is not None:
                    # headway = solver.compute_clcs_distance(self.cosy, ego_position, other_position)[0]
                    headway, _ = self._sd_distance(other_position, ego_position, ego_orientation)

                    if headway > 0:
                        veh_results.append(headway)
                        continue
                veh_results.append(np.inf)
            results[t] = min(veh_results)

        return results

    def thw(self):
        # see commonroad_crime.measure.time.thw for details
        results = pd.Series(None, index=list(range(self.t_start, self.t_end + 1)))
        for t in range(self.t_start, self.t_end + 1):
            if self.msg_logger is not None:
                self.msg_logger.debug(f"evaluating thw of {self.agent_id} at timestep {t}")
            ego_position = self.ego.state_at_time(t).position + self.ego.obstacle_shape.length / 2
            ego_s, ego_d = self.cosy.convert_to_curvilinear_coords(ego_position[0], ego_position[1])
            veh_results = []
            for other_obs in self.other_obstacles:
                other_position = self._rear_end_position(other_obs, t)
                if other_position is not None:
                    other_s, other_d = self.cosy.convert_to_curvilinear_coords(other_position[0], other_position[1])
                    if ego_s <= other_s:
                        reached = False
                        for ts in range(t + 1, self.t_end + 1):
                            ego_position = self.ego.state_at_time(ts).position + self.ego.obstacle_shape.length / 2
                            ego_s, ego_d = self.cosy.convert_to_curvilinear_coords(ego_position[0], ego_position[1])
                            if ego_s > other_s:
                                # trajectory reaches obstacle position at time step ts
                                veh_results.append((ts - t) * self.dt)
                                reached = True
                                break
                        if reached:
                            continue
                veh_results.append(np.inf)
                #         if not reached:
                #             veh_results.append(np.inf)
                #     if ego_s > other_s:
                #         # ego already ahead of other vehicle
                #         veh_results.append(np.inf)
                # else:
                #     veh_results.append(np.inf)

            results[t] = min(veh_results)

        return results

    def ttc(self):
        # see commonroad_crime.measure.time.ttc for details
        results = pd.Series(None, index=list(range(self.t_start, self.t_end + 1)))

        for t in range(self.t_start, self.t_end + 1):
            if self.msg_logger is not None:
                self.msg_logger.debug(f"evaluating ttc of {self.agent_id} at timestep {t}")
            ego_state = self.ego.state_at_time(t)
            ego_position = ego_state.position + self.ego.obstacle_shape.length / 2
            lanelet_id = self.scenario.lanelet_network.find_lanelet_by_position([ego_state.position])[0]
            ego_orientation = ego_state.orientation

            veh_results = []
            for other_obs in self.other_obstacles:
                other_position = self._rear_end_position(other_obs, t)
                if other_position is not None:
                    headway, _ = self._sd_distance(other_position, ego_position, ego_orientation)
                    if headway > 0:
                        # ego behind other vehicle
                        state_other = other_obs.state_at_time(t)
                        other_orientation = state_other.orientation #solver.compute_lanelet_width_orientation(
                            # self.scenario.lanelet_network.find_lanelet_by_id(lanelet_id[0]), state_other.position)[1]
                        # actual velocity and acceleration of both vehicles along the lanelet
                        try:
                            v_ego = (np.sign(ego_state.velocity)
                                     * np.sqrt(ego_state.velocity ** 2 + ego_state.velocity_y ** 2)
                                     * np.cos(ego_orientation)
                                     )
                        except AttributeError:
                            v_ego = ego_state.velocity
                        # include the directions
                        try:
                            a_ego = (np.sign(ego_state.acceleration)
                                     * np.sqrt(ego_state.acceleration ** 2 + ego_state.acceleration_y ** 2)
                                     * np.cos(ego_orientation)
                                     )
                        except AttributeError:
                            a_ego = ego_state.acceleration

                        if isinstance(other_obs, DynamicObstacle):
                            try:
                                v_other = (np.sqrt(state_other.velocity ** 2 + state_other.velocity_y ** 2)
                                           * np.cos(other_orientation))
                            except AttributeError:
                                v_other = state_other.velocity
                            try:
                                a_other = (np.sqrt(state_other.acceleration ** 2 + state_other.acceleration_y ** 2) *
                                           np.cos(other_orientation))
                            except AttributeError:
                                a_other = state_other.acceleration
                        else:
                            v_other = 0.0
                            a_other = 0.0
                        delta_v = v_other - v_ego
                        delta_a = a_other - a_ego

                        if delta_v < 0 and abs(delta_a) <= 0.1:
                            veh_results.append(-(headway / delta_v))
                        elif delta_v ** 2 - 2 * headway * delta_a < 0:
                            veh_results.append(np.inf)
                        elif (delta_v < 0 and delta_a != 0) or (delta_v >= 0 > delta_a):
                            first = -(delta_v / delta_a)
                            second = np.sqrt(delta_v ** 2 - 2 * headway * delta_a) / delta_a
                            veh_results.append(first - second)
                        else:  # delta_v >= 0 and delta_a >= 0
                            veh_results.append(np.inf)
                        continue
                veh_results.append(np.inf)
                #     if headway < 0:
                #         # ego in front of other vehicle
                #         veh_results.append(np.inf)
                # if not other_position:
                #     veh_results.append(np.inf)
            results[t] = min(veh_results)
        return results

    def jerk(self):
        if self.msg_logger is not None:
            self.msg_logger.debug(f"evaluating jerk of {self.agent_id}")
        results = pd.Series(None, index=list(range(self.t_start, self.t_end + 1)))
        try:
            jerk = [self.ego.state_at_time(t).jerk for t in range(self.t_start, self.t_end + 1)]
        except AttributeError:

            ego_acc = [self.ego.state_at_time(t).acceleration for t in range(self.t_start, self.t_end + 1)]
            jerk = np.gradient(ego_acc, self.dt)

        results = jerk
        return results

    def jerk_lat(self):
        if self.msg_logger is not None:
            self.msg_logger.debug(f"evaluating jerk_lat of {self.agent_id}")
        # see commonroad_crime.measure.jerk.lat_j for details
        results = pd.Series(None, index=list(range(self.t_start, self.t_end + 1)))
        jerk = self.jerk()
        for t in range(self.t_start, self.t_end + 1):
            ego_state = self.ego.state_at_time(t)
            ego_position = ego_state.position + self.ego.obstacle_shape.length / 2
            lanelet_id = self.scenario.lanelet_network.find_lanelet_by_position([ego_state.position])[0]
            ego_orientation = solver.compute_lanelet_width_orientation(
                self.scenario.lanelet_network.find_lanelet_by_id(lanelet_id[0]), ego_state.position)[1]
            results[t] = abs(jerk[t]*np.sin(ego_orientation))

        return results

    def jerk_long(self):
        if self.msg_logger is not None:
            self.msg_logger.debug(f"evaluating jerk_long of {self.agent_id}")
        # see commonroad_crime.measure.jerk.long_j for details
        results = pd.Series(None, index=list(range(self.t_start, self.t_end + 1)))
        jerk = self.jerk()
        for t in range(self.t_start, self.t_end + 1):
            ego_state = self.ego.state_at_time(t)
            ego_position = ego_state.position + self.ego.obstacle_shape.length / 2
            lanelet_id = self.scenario.lanelet_network.find_lanelet_by_position([ego_state.position])[0]
            ego_orientation = solver.compute_lanelet_width_orientation(
                self.scenario.lanelet_network.find_lanelet_by_id(lanelet_id[0]), ego_state.position)[1]
            results[t] = abs(jerk[t] * np.cos(ego_orientation))

        return results

    def coll_prob(self):
        pass

