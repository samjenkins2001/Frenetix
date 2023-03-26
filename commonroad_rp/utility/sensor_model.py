"""Sensor models for CommonRoad.

Author: Maximilian Geisslinger <maximilian.geisslinger@tum.de>
"""


# Standard imports
import os

# Thrird-party imports
import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from shapely.geometry import Point, Polygon
import shapely
import commonroad_rp.utility.helper_functions as hf


def get_visible_objects(
    scenario, time_step, ego_pos, ego_state, ego_id=42, sensor_radius=50, occlusions=True, wall_buffer=0.0
):
    """This function simulates a sensor model of a camera/lidar sensor.

    It returns the visible objects and the visible area.

    Arguments:
        scenario {[CommonRoad scenario object]} -- [Commonroad Scenario]
        time_step {[int]} -- [time step for commonroad scenario]
        ego_pos {[list]} -- [list with x and y coordinates of ego position]

    Keyword Arguments:
        sensor_radius {int} -- [description] (default: {50})
        occlusion {bool} -- [True if occlusions by dynamic obstacles should be considered] (default: {True})
        wall_buffer {float} -- [Buffer for visibility around corners in meters] (default: {0.0})


    Returns:
        visible_object_ids [list] -- [list of objects that are visible]
        visible_area [shapely object] -- [area that is visible (for visualization e.g.)]

    """

    # Create circle from sensor radius
    visible_area = Point(ego_pos).buffer(sensor_radius)

    # Reduce visible area to lanelets
    for idx, lnlet in enumerate(scenario.lanelet_network.lanelets):
        pol_vertices = Polygon(
            np.concatenate((lnlet.right_vertices, lnlet.left_vertices[::-1]))
        )
        if not pol_vertices.is_valid:
            continue

        visible_lnlet = visible_area.intersection(pol_vertices)

        if idx == 0:
            new_vis_area = visible_lnlet
        else:
            new_vis_area = new_vis_area.union(visible_lnlet)

    visible_area = new_vis_area

    # Enlarge visible area by wall buffer
    visible_area = visible_area.buffer(wall_buffer)

    # Substract areas that can not be seen due to geometry
    if visible_area.geom_type == 'MultiPolygon':
        allparts = [p.buffer(0) for p in visible_area.geometry]
        visible_area.geometry = shapely.ops.cascaded_union(allparts)

    points_vis_area = np.array(visible_area.exterior.xy).T

    for idx in range(points_vis_area.shape[0] - 1):
        vert_point1 = points_vis_area[idx]
        vert_point2 = points_vis_area[idx + 1]

        pol = _create_polygon_from_vertices(vert_point1, vert_point2, ego_pos)

        if pol.is_valid:
            visible_area = visible_area.difference(pol)

    # if occlusions through dynamic objects should be considered
    if occlusions:

        for obst in scenario.obstacles:
            # check if obstacle is still there
            try:
                if obst.obstacle_role.name == "STATIC":
                    pos = obst.initial_state.position
                    orientation = obst.initial_state.orientation
                else:
                    pos = obst.prediction.trajectory.state_list[time_step].position
                    orientation = obst.prediction.trajectory.state_list[
                        time_step
                    ].orientation
            except IndexError:
                continue

            pos_point = Point(pos)
            # check if within sensor radius
            if pos_point.within(visible_area):
                # Substract occlusions from dynamic obstacles
                # Calculate corner points in world coordinates
                corner_points = _calc_corner_points(
                    pos, orientation, obst.obstacle_shape
                )

                # Identify points for geometric projection
                r1, r2 = _identify_projection_points(corner_points, ego_pos)

                # Create polygon with points far away in the ray direction of ego pos
                r3 = r2 + __unit_vector(r2 - ego_pos) * sensor_radius
                r4 = r1 + __unit_vector(r1 - ego_pos) * sensor_radius

                occlusion = Polygon([r1, r2, r3, r4])

                # Substract occlusion from visible area
                if occlusion.is_valid:
                    visible_area = visible_area.difference(occlusion)

    # Get visible objects
    visible_object_ids = []

    for obst in scenario.obstacles:
        # check if obstacle is still there
        try:
            if obst.obstacle_role.name == "STATIC":
                pos = obst.initial_state.position
                orientation = obst.initial_state.orientation
            else:
                pos = obst.prediction.trajectory.state_list[time_step].position
                orientation = obst.prediction.trajectory.state_list[
                    time_step
                ].orientation

        except IndexError:
            continue

        corner_points = _calc_corner_points(pos, orientation, obst.obstacle_shape)
        dyn_obst_shape = Polygon(corner_points)

        if dyn_obst_shape.intersects(visible_area):
            visible_object_ids.append(obst.obstacle_id)

    # # Add static obstacles
    # obstacles_within_radius = []
    # for obstacle in scenario.static_obstacles:
    #     # do not consider the ego vehicle
    #     if obstacle.obstacle_id != ego_id:
    #         occ = obstacle.occupancy_at_time(ego_state.time_step)
    #         # if the obstacle is not in the lanelet network at the given time, its occupancy is None
    #         if occ is not None:
    #             # calculate the distance between the two obstacles
    #             dist = hf.distance(
    #                 pos1=ego_state.position,
    #                 pos2=obstacle.occupancy_at_time(ego_state.time_step).shape.center,
    #             )
    #             # add obstacles that are close enough
    #             if dist < sensor_radius:
    #                 obstacles_within_radius.append(obstacle.obstacle_id)

    return visible_object_ids, visible_area


def _calc_corner_points(pos, orientation, obstacle_shape):
    """Calculate corner points of a dynamic obstacles in global coordinate system.

    Arguments:
        pos {[type]} -- [description]
        orientation {[type]} -- [description]
        obstacle_shape {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    corner_points = _rotate_point_by_angle(obstacle_shape.vertices[0:4], orientation)
    corner_points = [p + pos for p in corner_points]
    return np.array(corner_points)


def _rotate_point_by_angle(point, angle):
    """Rotate any point by an angle.

    Arguments:
        point {[type]} -- [description]
        angle {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return np.matmul(rotation_matrix, point.transpose()).transpose()


def _identify_projection_points(corner_points, ego_pos):
    """This function identifies the two points of an rectangular objects that are the edges from an ego pos point of view.

    Arguments:
        corner_points {[type]} -- [description]
        ego_pos {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    max_angle = 0

    for edge_point1 in corner_points:
        for edge_point2 in corner_points:
            ray1 = edge_point1 - ego_pos
            ray2 = edge_point2 - ego_pos

            angle = _angle_between(ray1, ray2)

            if angle > max_angle:
                max_angle = angle
                ret_edge_point1 = edge_point1
                ret_edge_point2 = edge_point2

    return ret_edge_point1, ret_edge_point2


def _create_polygon_from_vertices(vert_point1, vert_point2, ego_pos):
    """Creates a polygon for the area that is occluded from two vertice points.

    Arguments:
        vert_point1 {[list]} -- [x,y of first point of object]
        vert_point2 {[list]} -- [x,y of second point of object]
        ego_pos {[list]} -- [x,y of ego position]

    Returns:
        pol [Shapely polygon] -- [Represents the occluded area]
    """

    pol_point1 = vert_point1 + 100 * (vert_point1 - ego_pos)
    pol_point2 = vert_point2 + 100 * (vert_point2 - ego_pos)

    pol = Polygon([vert_point1, vert_point2, pol_point2, pol_point1, vert_point1])

    return pol


def __unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def _angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2':"""
    v1_u = __unit_vector(v1)
    v2_u = __unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def clear_console():
    """Clear console."""
    # for windows
    if os.name == 'nt':
        os.system('cls')

    # for mac and linux(here, os.name is 'posix')
    else:
        os.system('clear')


if __name__ == '__main__':
    # Custom imports
    from visualization import visualize_timestep

    # Set TERM variable
    os.environ["TERM"] = "xterm"

    scenario_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'commonroad-scenarios/scenarios/recorded/NGSIM/Lankershim/USA_Lanker-2_16_T-1.xml',
    )
    scenario, planning_problem_set = CommonRoadFileReader(scenario_path).open()

    # Initial position
    problem_ids = list(planning_problem_set.planning_problem_dict.keys())
    ego_pos = planning_problem_set.planning_problem_dict[
        problem_ids[0]
    ].initial_state.position

    time_step = 0

    while time_step < 200:
        visible_object_ids, visible_area = get_visible_objects(
            scenario, time_step, ego_pos
        )

        clear_console()
        print("Visible object IDs: {}".format(visible_object_ids), end='\r')

        visualize_timestep(
            scenario, time_step, visible_area=visible_area, save_animation=False
        )

        time_step += 1
