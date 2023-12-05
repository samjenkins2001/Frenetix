from commonroad.geometry.shape import Polygon, Circle
#mfrom commonroad_crime.utility import solver, general
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
import pandas as pd
import numpy as np
from risk_assessment.risk_costs import calc_risk

# def evaluation_wrapper(args):
#     agent_id, time_step, used_measures, config_measures = args
#     return evaluate_timestep(agent_id, time_step, used_measures, config_measures)
class Measures:
    # adapted from CriMe
    def __init__(self, agent_id, scenario, t_start, t_end, ref_path=None, msg_logger = None):
        self.agent_id = agent_id
        self.scenario = scenario
        self.dt = scenario.dt
        self.t_start = t_start
        self.t_end = t_end
        self.ego = scenario.obstacle_by_id(agent_id)
        self.other_obstacles = self._set_other_obstacles()
        self.cosy = self._update_clcs(ref_path)
        self.msg_logger = msg_logger

    def _set_other_obstacles(self):
        other_obstacles = [obs for obs in self.scenario.obstacles
                           if obs.obstacle_id is not self.agent_id]
        return other_obstacles

    def _update_clcs(self, ref_path):
        """
        Updates the curvilinear coordinate system in the configuration setting using the reference path from the lanelet
        where the ego vehicle is currently located to the end of the lanelet.
        """
        # default setting of ego vehicle's curvilinear coordinate system
        if not ref_path:
            ego_initial_lanelet_id = list(
                self.ego.prediction.center_lanelet_assignment[0]
            )[0]
            ref_path = general.generate_reference_path(ego_initial_lanelet_id, self.scenario.lanelet_network)
        ccs = CurvilinearCoordinateSystem(ref_path)
        return ccs

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
            veh_results = []
            for other_obs in self.other_obstacles:
                other_position = self._rear_end_position(other_obs, t)
                if other_position is not None:
                    headway = solver.compute_clcs_distance(self.cosy, ego_position, other_position)[0]
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
            ego_orientation = solver.compute_lanelet_width_orientation(
                self.scenario.lanelet_network.find_lanelet_by_id(lanelet_id[0]), ego_state.position)[1]

            veh_results = []
            for other_obs in self.other_obstacles:
                other_position = self._rear_end_position(other_obs, t)
                if other_position is not None:
                    headway = solver.compute_clcs_distance(self.cosy, ego_position, other_position)[0]
                    if headway > 0:
                        # ego behind other vehicle
                        state_other = other_obs.state_at_time(t)
                        other_orientation = solver.compute_lanelet_width_orientation(
                            self.scenario.lanelet_network.find_lanelet_by_id(lanelet_id[0]), state_other.position)[1]
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

