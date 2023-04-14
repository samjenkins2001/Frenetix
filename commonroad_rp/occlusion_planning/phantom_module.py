# imports
import numpy as np
from scipy.spatial import distance
from shapely.geometry import Point, LineString
from typing import Tuple

# commonroad imports
import commonroad_rp.occlusion_planning.utils.occ_helper_functions as ohf
import commonroad_rp.occlusion_planning.utils.vis_helper_functions as vhf
from commonroad_rp.occlusion_planning.occlusion_obstacles import OccPhantomObstacle

# risk assessment imports
from risk_assessment.helpers.collision_helper_function import create_tvobstacle

"""
This module is the main module of the Phantom obstacle generation of the reactive planner. 

Author: Korbinian Moller, TUM
Date: 14.04.2023
"""


class OccPhantomModule:
    def __init__(self, config=None, occ_scenario=None, vis_module=None,
                 occ_visible_area=None, occ_map=None, occ_plot=None):
        self.config = config.occlusion
        self.occ_scenario = occ_scenario
        self.vis_module = vis_module
        self.occ_visible_area = occ_visible_area
        self.occ_map = occ_map
        self.occ_plot = occ_plot
        self.max_number_of_phantom_peds = 1
        self.ped_width = 0.5
        self.ped_length = 0.3
        self.ped_velocity = 0.8
        self.sorted_static_obstacles = None
        self.phantom_peds = None
        self.ego_pos_s = None
        self.max_s_trajectories = None

        # calculate the maximum distance from a vehicle corner point to the vehicle center point
        self.ego_diag = np.sqrt(
            np.power(self.occ_scenario.ego_length, 2) + np.power(self.occ_scenario.ego_width, 2))

    def evaluate_trajectories(self, trajectories):

        # calculate spawn points and create phantom obstacles
        self._evaluation_preprocessing(trajectories)
        collisions = []

        # if no phantom obstacles exist
        if self.phantom_peds is None or len(self.phantom_peds) == 0:
            return

        # iterate over trajectories
        for traj in trajectories:
            # check if trajectory leads to a collision with phantom obstacle -> dict
            collision = self._check_trajectory_collision(traj, mode=self.config.collision_check_mode)

            if collision['collision']:
                cts = collision['collision_timestep']
                ped = collision['ped']

                # collision['scenario'] =
                # collision['vehicle_params'] =
                collision['ego_velocity'] = traj.cartesian.v[cts]
                # collision['obstacle_id'] =
                collision['obstacle_size'] = ped.width * ped.length
                collision['obstacle_velocity'] = ped.v
                # collision['modes'] =
                # collision['coeffs'] =

                # calculate angles that are needed for harm model
                angles = ohf.calc_collision_angles(ped, traj, cts)

                # add angles to collision dict
                collision.update(angles)

                # visualize collision of trajectory
                if self.config.visualize_collision:
                    self.occ_plot.plot_phantom_collision(collision)

            collisions.append(collision)

        # plot phantom peds and their trajectories if plot is activated
        if self.occ_plot is not None:
            self._plot_phantom()

        return collisions

    def _check_trajectory_collision(self, traj, mode='shapely') -> dict:
        """
        Checks whether a trajectory leads to a collision with a phantom pedestrian in the scenario.
        Args:
            traj: commonroad trajectory sample
            mode: mode whether to use shapely or Commonroad pycrcc

        Returns:
            dict with collision information
        """
        # check if mode is valid
        valid_modes = ['shapely', 'pycrcc']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Valid modes are: {', '.join(valid_modes)}")

        # init ego trajectory polygons
        ego_traj_polygons = None

        # init collision dict
        collision_dict = {'collision': False, 'collision_timestep': None, 'ped_id': None, 'ped': None,
                          'traj': traj, 'ego_traj_polygons': None}

        # iterate over phantom peds
        for ped in self.phantom_peds:

            # check for collision using pycrcc
            if mode == 'pycrcc':
                for i in range(len(traj.cartesian.x)):

                    # create time variant obstacle for ego
                    ego_tvo = create_tvobstacle(traj_list=[[traj.cartesian.x[i], traj.cartesian.y[i],
                                                            traj.cartesian.theta[i]]],
                                                box_length=self.occ_scenario.ego_length / 2,
                                                box_width=self.occ_scenario.ego_width / 2,
                                                start_time_step=i)

                    # check for collision at timestep i
                    collision_cr = ego_tvo.collide(ped.cr_collision_object)

                    # if collision is true, update collision_dict and return
                    if collision_cr is True:
                        collision_dict = {'collision': True, 'collision_timestep': i,
                                          'ped_id': ped.obstacle_id, 'ped': ped, 'traj': traj,
                                          'ego_traj_polygons': None}
                        return collision_dict

            # check for collisions using shapely
            elif mode == 'shapely':
                # check if collision is possible (s coordinates match) and perform detailed collision check
                if np.any((traj.curvilinear.s + self.ego_diag / 2) >= (ped.s - ped.diag / 2)):

                    # if trajectory has not been converted to polygon list, create polygon list
                    if ego_traj_polygons is None:
                        ego_traj_polygons = ohf.compute_vehicle_polygons(traj.cartesian.x, traj.cartesian.y,
                                                                         traj.cartesian.theta,
                                                                         self.occ_scenario.ego_width,
                                                                         self.occ_scenario.ego_length)

                    # store polygon list in local variable and shorten it
                    ped_traj_polygons = ped.traj_polygons['polygons'][:len(ego_traj_polygons['polygons'])]

                    # iterate through polygons and check for collision
                    for i, (ego_poly, ped_poly) in enumerate(zip(ego_traj_polygons['polygons'], ped_traj_polygons)):

                        if ego_poly.intersects(ped_poly):
                            collision_dict = {'collision': True, 'collision_timestep': i,
                                              'ped_id': ped.obstacle_id, 'ped': ped, 'traj': traj,
                                              'ego_traj_polygons': ego_traj_polygons}

                            return collision_dict

        return collision_dict

    def _evaluation_preprocessing(self, trajectories):

        # only consider static obstacles (ped won't be in front of a moving car)
        self.sorted_static_obstacles = [obst for obst in self.vis_module.obstacles if obst.obstacle_role == 'STATIC']

        # quit here, if no static obstacle exists
        if len(self.sorted_static_obstacles) == 0:
            return

        # sort obstacles by distance to ego pos
        self.sorted_static_obstacles = sorted(self.sorted_static_obstacles, key=lambda obstacle:
                                              distance.euclidean(obstacle.pos, self.vis_module.ego_pos))

        # find max s in trajectories
        self.max_s_trajectories = ohf.find_max_s_in_trajectories(trajectories)

        # check, which obstacles area along the driving direction
        self._check_obstacles()

        # find spawn points (dict with xy and cl) and points on ref path for phantom pedestrians
        spawn_points, points_on_ref_path = self._find_spawn_points()

        # create phantom pedestrians
        self._create_phantom_peds(spawn_points, points_on_ref_path)

        # calculate trajectories of phantom peds
        self._calc_trajectories()

    def _plot_phantom(self):
        ax = self.occ_plot.ax
        for ped in self.phantom_peds:
            ohf.fill_polygons(ax, ped.polygon, 'gold', zorder=10)
            ax.plot(ped.trajectory[:, 0], ped.trajectory[:, 1], 'b')
            ax.plot(ped.goal_position[0], ped.goal_position[1], 'bo')

        self.occ_plot._save_plot()

    def _create_phantom_peds(self, spawn_points, points_on_ref_path):
        # create empty list for phantom pedestrians
        phantom_peds = []

        # iterate over spawn points
        for i, spawn_point in enumerate(spawn_points):
            # create vector to calculate orientation
            vector = points_on_ref_path[i] - spawn_points[i]['xy']

            # calculate orientation with vector and reference vector [1, 0]
            orientation = vhf.angle_between(np.array([1, 0]), vector)

            # find needed parameters for phantom ped
            create_cr_obst = False
            calc_ped_traj_polygons = False

            if self.config.visualize_collision or self.config.collision_check_mode == 'shapely':
                calc_ped_traj_polygons = True

            if self.config.create_commonroad_obstacle or self.config.collision_check_mode == 'pycrcc':
                create_cr_obst = True

            # create phantom pedestrian and add to list
            phantom_peds.append(OccPhantomObstacle(i + 1, spawn_point['xy'], orientation, self.ped_length,
                                                   self.ped_width, vector, s=spawn_point['cl'][0],
                                                   calc_ped_traj_polygons=calc_ped_traj_polygons,
                                                   create_cr_obst=create_cr_obst))

        # assign local variable to object
        self.phantom_peds = phantom_peds

    def _calc_trajectories(self):
        for ped in self.phantom_peds:
            ped.set_velocity(self.ped_velocity)

            ped.calc_goal_position(self.occ_scenario.sidewalk_combined)
            ped.calc_trajectory(dt=self.occ_scenario.dt)

    def _check_obstacles(self):

        for obst in self.sorted_static_obstacles:
            if obst.polygon.intersects(self.occ_visible_area.poly.buffer(0.1)):
                obst.visible_in_driving_direction = True
            else:
                obst.visible_in_driving_direction = False

    def _find_spawn_points(self, shift_spawn_points=True):

        self.ego_pos_s = self.occ_scenario.ref_path_ls.project(Point(self.vis_module.ego_pos))
        spawn_points = []
        spawn_points_shifted = []
        points_on_ref_path = []

        if self.vis_module.time_step == 15:
            print('asd')

        # iterate over obstacles (sorted by distance to ego position)
        for obst in self.sorted_static_obstacles:

            # if the obstacle is visible (and not behind the vehicle), find relevant corner points
            if obst.visible_in_driving_direction:

                # init dict to store xy and cl coordinates
                relevant_cp = dict()
                spawn_point = dict()

                # find relevant corner points (cp) for each obstacle and the corresponding vectors
                # to ref path (needed later)
                relevant_cp['xy'], point_on_rp = self._spawn_point_preprocessing(obst, offset=self.ped_width / 2 * 1.1)

                # calc curvilinear coordinates for relevant_cp and remove points that are too far away
                relevant_cp['cl'], relevant_cp['xy'], point_on_rp = \
                    self._convert_to_cl_and_remove_far_points(relevant_cp['xy'], point_on_rp)

                # check if more than one possible spawn point exists for each obstacle
                if len(relevant_cp['xy']) > 1:

                    # if more than one relevant corner point exists, find critical spawn point and vector_to_ref_path
                    spawn_point['xy'], spawn_point['cl'], point_on_rp = \
                        self._find_relevant_spawn_point(relevant_cp['cl'], relevant_cp['xy'], point_on_rp)

                    # add spawn point to spawn_points array
                    spawn_points.append(spawn_point)

                    # add vector_to_ref_path to array
                    points_on_ref_path.append(point_on_rp)
                elif len(relevant_cp['xy']) == 1:
                    # only one relevant corner exists -> add relevant corner to spawn_points array

                    spawn_point['xy'] = np.array(relevant_cp['xy'][0])
                    spawn_point['cl'] = np.array(relevant_cp['cl'][0])

                    spawn_points.append(spawn_point)

                    # add vector_to_ref_path to array
                    points_on_ref_path.append(point_on_rp[0])

                else:
                    continue

        # limit spawn points to max_number_of_phantom_peds
        if len(spawn_points) > self.max_number_of_phantom_peds:
            # sort spawn points according to their distance to the ego position
            sorted_spawn_points_with_index = sorted(enumerate(spawn_points), key=lambda sp:
                                                    distance.euclidean(sp[1]['xy'], self.vis_module.ego_pos))

            # convert combined list to separate lists
            sorted_indices, sorted_spawn_points = zip(*sorted_spawn_points_with_index)

            # reorder list of points_on_ref_path according to the spawn points
            sorted_points_on_ref_path = [points_on_ref_path[i] for i in sorted_indices]

            # select relevant points
            spawn_points = sorted_spawn_points[:self.max_number_of_phantom_peds]
            points_on_ref_path = sorted_points_on_ref_path[:self.max_number_of_phantom_peds]

        if shift_spawn_points and len(spawn_points) > 0:
            ref_path_ls = self.occ_scenario.ref_path_ls
            for i, spawn_point in enumerate(spawn_points):
                line = np.stack([spawn_points[0]['xy'], points_on_ref_path[i]])
                line_ls = LineString(line)

                spawn_point_shifted_point = self.vis_module.visible_area_timestep.exterior.intersection(line_ls)

                if spawn_point_shifted_point.is_empty:
                    # append dict to spawn_points_shifted
                    spawn_points_shifted.append(spawn_point)
                else:
                    spawn_point_shifted = dict()
                    spawn_point_shifted['xy'] = np.array(spawn_point_shifted_point.coords).flatten()
                    s = ref_path_ls.project(spawn_point_shifted_point)
                    d = ref_path_ls.distance(spawn_point_shifted_point)
                    spawn_point_shifted['cl'] = np.array([s, d])

                    spawn_points_shifted.append(spawn_point_shifted)

            spawn_points = spawn_points_shifted

        return spawn_points, points_on_ref_path

    def _convert_to_cl_and_remove_far_points(self, relevant_cp_xy, points_on_rp):
        # initialize variables
        ref_path_ls = self.occ_scenario.ref_path_ls
        relevant_cp_cl = []
        relevant_cp_xy_new = []
        points_on_rp_new = []

        # iterate through corner points and calculate curvilinear s and d coordinate for each corner
        for i, corner in enumerate(relevant_cp_xy):
            s = ref_path_ls.project(Point(corner))
            d = ref_path_ls.distance(Point(corner))

            # only add points that can be reached by a trajectory
            if not s >= self.max_s_trajectories and not s < self.ego_pos_s:
                relevant_cp_cl.append((s, d))
                relevant_cp_xy_new.append(corner)
                points_on_rp_new.append(points_on_rp[i])

        return relevant_cp_cl, relevant_cp_xy_new, points_on_rp_new

    def _find_relevant_spawn_point(self, relevant_cp_cl, relevant_cp_xy, point_on_ref_path):
        """
        This function uses the list of relevant corner points and finds the most critical spawn point for a phantom
        pedestrian. Only one spawn point will be determined for each obstacle.
        Args:
            relevant_cp_cl: list of relevant corner points in curvilinear (s,d) coordinates
            relevant_cp_xy: list of relevant corner points in world (x,y) coordinates

        Returns: numpy array of spawn point

        """
        # convert list of curvilinear coordinates to numpy array [s,d]
        relevant_cp_cl = np.array(relevant_cp_cl)

        # threshold, that points are "close" (only s considered)
        threshold = 0.2

        # calc distance of s coordinates between corner_cl and ego pos (projected on ref path)
        distances = abs(relevant_cp_cl[:, 0] - self.ego_pos_s)

        # find index of the closest point to ego pos (only s considered)
        min_distance_index = np.argmin(distances)

        # check whether there is a point, that has a similar s coordinate
        similar_indices = np.where(np.isclose(distances, distances[min_distance_index], rtol=0, atol=threshold))[0]

        # if two points have similar s coordinates, the point with the smaller d coordinate shall be used
        if len(similar_indices) > 1:
            # find point with smaller d coordinate
            min_d_index = np.argmin(relevant_cp_cl[similar_indices, 1])

            # update index
            min_distance_index = similar_indices[min_d_index]

        # assign spawn point
        spawn_point_xy = np.array(relevant_cp_xy[min_distance_index])
        spawn_point_cl = np.array(relevant_cp_cl[min_distance_index])

        # select corresponding vector to ref path
        point_on_rp_xy = point_on_ref_path[min_distance_index]

        return spawn_point_xy, spawn_point_cl, point_on_rp_xy

    def _spawn_point_preprocessing(self, obst, offset=0.25) -> Tuple[list, list]:
        """
        This function calculates relevant corner points and returns them as a list (for each obstacle)
        Args:
            obst: obstacle (one obstacle of type OcclusionObstacle)

        Returns: - list of relevant corner points (potential spawn points for phantom pedestrians)
                 - point on ref path (needed later)

        """
        # load ref path linestring from occ scenario
        ref_path_ls = self.occ_scenario.ref_path_ls

        # initialize list for relevant corner points
        relevant_corner_points = []
        point_on_rp = []

        # if offset shall be used, calculate corner points from polygon, otherwise use corner points of obstacle
        if offset > 0:
            corners = obst.polygon.buffer(offset, join_style=2).exterior.coords[:-1]
        else:
            corners = obst.corner_points

        # iterate over corner points
        for corner in corners:

            # prepare needed variables to calculate points and vectors
            cp = Point(corner)
            point_on_ref_path = np.array(ref_path_ls.interpolate(ref_path_ls.project(cp)).coords).flatten()
            vector = point_on_ref_path - corner
            normal_vector = ohf.unit_vector(vector)

            # calc points towards and in the opposite direction to ref path
            point_to_rp = corner + 0.1 * normal_vector
            point_opp_rp = corner - 0.1 * normal_vector

            # if the opposite point is within the visible area, the corner is not relevant
            if Point(point_opp_rp).within(self.vis_module.visible_area_timestep.buffer(0.01)):
                continue

            # if the opposite point is within the obstacle, the corner is not relevant
            if Point(point_opp_rp).within(obst.polygon.buffer(-0.01)):
                continue

            # if the point towards the ref path is within the obstacle, the corner is not relevant
            if Point(point_to_rp).within(obst.polygon.buffer(-0.01)):
                continue

            # if the point is outside the lanelet network (including sidewalk), the corner is not relevant
            if not Point(point_opp_rp).within(self.occ_scenario.lanelets_combined_with_sidewalk):
                continue

            # check if phantom pedestrian would be within another obstacle
            if cp.buffer(self.ped_width / 2).intersects(self.vis_module.obstacles_polygon):
                continue

            # if corner point is a potential spawn point for phantom peds, add to list
            relevant_corner_points.append(corner)
            point_on_rp.append(point_on_ref_path)

        return relevant_corner_points, point_on_rp
