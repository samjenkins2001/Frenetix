import numpy as np
import commonroad_rp.occlusion_planning.utils.occ_helper_functions as ohf
import commonroad_rp.occlusion_planning.utils.vis_helper_functions as vhf
from shapely.geometry import Point, Polygon
from commonroad.geometry.shape import Rectangle


class OcclusionObstacle:
    def __init__(self, obst):
        self.obstacle_id = obst.obstacle_id
        self.obstacle_role = obst.obstacle_role.name
        self.obstacle_shape = obst.obstacle_shape
        self.visible_at_timestep = False
        self.relevant_corner_points = None

        if self.obstacle_role == "STATIC":
            self.pos = obst.initial_state.position
            self.orientation = obst.initial_state.orientation
            self.corner_points = vhf.calc_corner_points(self.pos, self.orientation, self.obstacle_shape)
            self.polygon = Polygon(self.corner_points)
            self.pos_point = Point(self.pos)
        else:
            self.pos = [state.position for state in obst.prediction.trajectory.state_list]
            self.orientation = [state.orientation for state in obst.prediction.trajectory.state_list]
            # corner points and polygon will be updated each timestep, no history saved
            self.corner_points = vhf.calc_corner_points(self.pos[0], self.orientation[0], self.obstacle_shape)
            self.polygon = Polygon(self.corner_points)
            self.pos_point = Point(self.pos[0])

    def update_at_timestep(self, time_step):
        if not self.obstacle_role == "STATIC" and self.pos is not None:
            try:
                self.corner_points = vhf.calc_corner_points(self.pos[time_step], self.orientation[time_step],
                                                            self.obstacle_shape)
                self.polygon = Polygon(self.corner_points)
                self.pos_point = Point(self.pos[time_step])
            except IndexError:
                self.pos = None
                self.orientation = None
                self.corner_points = None
                self.polygon = None
                self.pos_point = None

        self.relevant_corner_points = None
        self.visible_at_timestep = False

    def set_relevant_corner_points(self, c1, c2):
        self.relevant_corner_points = np.array([c1, c2])


class OccBasicObstacle:
    """
    OccBasicObstacles are similar to OcclusionObstacles, but contain less information. They are the super class to
    EstimationObstacle and PhantomObstacle.
    """
    def __init__(self, obst_id, pos, orientation, length, width):
        self.obstacle_id = obst_id
        self.pos = pos
        self.orientation = orientation
        self.length = length
        self.width = width

        # create rectangle object from commonroad.geometry.shape (vertices are needed for further calculation)
        self.shape = Rectangle(length, width, center=np.array([0.0, 0.0]), orientation=0.0)

        self.corner_points = vhf.calc_corner_points(self.pos, self.orientation, self.shape)
        self.polygon = Polygon(self.corner_points)
        self.pos_point = Point(self.pos)


class EstimationObstacle(OccBasicObstacle):
    """
    obstacles are essential for accurately estimating visible and occluded areas.
    These obstacles are represented at their predicted positions at timestep x. These obstacles have to be created,
    so that the already implemented function calc_visible_area_from_obstacle_occlusions can be reused.
    """

    def __init__(self, obst_id, prediction, timestep):
        pos = prediction['pos_list'][timestep]
        orientation = prediction['orientation_list'][timestep]
        length = prediction['shape']['length']
        width = prediction['shape']['width']

        super().__init__(obst_id, pos, orientation, length, width)


class PhantomObstacle(OccBasicObstacle):
    def __init__(self, phantom_id, pos, orientation, length=0.5, width=1):

        super().__init__(phantom_id, pos, orientation, length, width)

