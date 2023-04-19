import numpy as np
import commonroad_rp.occlusion_planning.utils.vis_helper_functions as vhf
import commonroad_rp.occlusion_planning.utils.occ_helper_functions as ohf
from commonroad_rp.occlusion_planning.occlusion_obstacles import OcclusionObstacle


class VisibilityModule:
    def __init__(self, scenario=None, lanelets=None, sensor_radius=50, occlusions=True, wall_buffer=0.0):
        self.ego_state = None
        self.lanelets = lanelets
        self.sensor_radius = sensor_radius
        self.occlusions = occlusions
        self.wall_buffer = wall_buffer
        self.obstacles = convert_obstacles(scenario.obstacles)
        self.ego_pos = None
        self.time_step = None
        self.visible_area_timestep = None
        self.visible_objects_timestep = None
        self.obstacles_polygon = None

    def update_obstacles_at_time_step(self, time_step):
        for obst in self.obstacles:
            obst.update_at_timestep(time_step)

    def get_visible_area_and_objects(self, ego_state=None):

        if ego_state is None:
            raise ValueError("Ego state must be provided for the calculation of visible area and visible objects!")

        # initialize ego position and current time step
        self.ego_state = ego_state
        self.ego_pos = ego_state.position
        self.time_step = ego_state.time_step

        # initialize list to store visible obstacles
        visible_objects_timestep = []

        # update corner points and polygons for dynamic obstacles
        if self.time_step > 0:  # initial values are already set
            self.update_obstacles_at_time_step(self.time_step)

        # calculate visible area only considering the lanelet geometry
        visible_area = vhf.calc_visible_area_from_lanelet_geometry(self.lanelets, self.ego_pos, self.sensor_radius)

        # if obstacle occlusions shall be considered, subtract polygon from visible_area
        if self.occlusions:

            # update visible_area due to obstacle occlusion
            visible_area, obst_polygon = vhf.calc_visible_area_from_obstacle_occlusions(visible_area, self.ego_pos,
                                                                                        self.obstacles,
                                                                                        self.sensor_radius,
                                                                                        return_shapely_obstacles=True)

            # assign multipolygon of all obstacles to variable (needed for phantom pedestrian calculation)
            self.obstacles_polygon = obst_polygon

        # get visible obstacles and add to list
        visible_area_check = visible_area.buffer(0.01, join_style=2)
        for obst in self.obstacles:
            if obst.pos is not None:
                if obst.polygon.intersects(visible_area_check):
                    visible_objects_timestep.append(obst.obstacle_id)
                    obst.visible_at_timestep = True

        # remove linestrings from visible_area
        visible_area = ohf.remove_unwanted_shapely_elements(visible_area)

        # save visible_objects and visible area in VisibilityModule object
        self.visible_objects_timestep = visible_objects_timestep
        self.visible_area_timestep = visible_area

        return visible_objects_timestep, visible_area


def convert_obstacles(obstacles):
    occ_obstacles = []
    for obst in obstacles:
        occ_obst = OcclusionObstacle(obst)
        occ_obstacles.append(occ_obst)
    return occ_obstacles

