import numpy as np
import commonroad_rp.occlusion_planning.utils.occ_helper_functions as ohf
import commonroad_rp.occlusion_planning.utils.vis_helper_functions as vhf
from shapely.geometry import Point, Polygon, LineString
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType, PhantomObstacle
from commonroad.scenario.state import InitialState, CustomState
from commonroad.prediction.prediction import TrajectoryPrediction, SetBasedPrediction
from commonroad.scenario.trajectory import Trajectory
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object


class OcclusionObstacle:
    def __init__(self, obst):
        self.obstacle_id = obst.obstacle_id
        self.obstacle_role = obst.obstacle_role.name
        self.obstacle_shape = obst.obstacle_shape
        self.visible_at_timestep = False
        self.visible_in_driving_direction = None

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

        self.visible_at_timestep = False

        # needed for phantom pedestrian creation - will be calculated in OccPhantomModule
        self.visible_in_driving_direction = None


class OccBasicObstacle:
    """
    OccBasicObstacles are similar to OcclusionObstacles, but contain less information. They are the super class to
    EstimationObstacle and OccPhantomObstacle.
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


class OccPhantomObstacle(OccBasicObstacle):
    def __init__(self, phantom_id, pos, orientation, length=0.3, width=0.5, vector=None, s=None,
                 calc_ped_traj_polygons=False, create_cr_obst=False):

        # initialize phantom obstacle using OccBasicObstacle class
        super().__init__(phantom_id, pos, orientation, length, width)

        # assign additional variables
        self.calc_ped_traj_polygons = calc_ped_traj_polygons
        self.create_cr_obst = create_cr_obst
        self.orientations = []
        self.orientation_vector = vector
        self.v = 0
        self.trajectory = None
        self.goal_position = None
        self.trajectory_length = None
        self.traj_polygons = None
        self.s = s
        self.diag = np.sqrt(np.power(length, 2) + np.power(width, 2))
        self.commonroad_dynamic_obstacle = None
        self.commonroad_phantom_obstacle = None
        self.cr_collision_object = None

    def _create_cr_collision_object(self):
        self.cr_collision_object = create_collision_object(self.commonroad_dynamic_obstacle)

    def _create_cr_obstacle(self):

        initial_state = InitialState(time_step=0,
                                     position=self.pos,
                                     orientation=self.orientation,
                                     velocity=self.v,
                                     acceleration=0.0,
                                     yaw_rate=0.0,
                                     slip_angle=0.0)

        state_list = []

        for i in range(0, len(self.trajectory)):
            custom_state = CustomState(orientation=self.orientation,
                                       velocity=self.v,
                                       position=self.trajectory[i],
                                       time_step=i)

            state_list.append(custom_state)

        trajectory = Trajectory(initial_time_step=0, state_list=state_list)

        trajectory_prediction = TrajectoryPrediction(trajectory=trajectory,
                                                     shape=self.shape)

        occupancy_set = trajectory_prediction.occupancy_set

        set_based_prediction = SetBasedPrediction(initial_time_step=0,
                                                  occupancy_set=occupancy_set)

        self.commonroad_dynamic_obstacle = DynamicObstacle(obstacle_id=self.obstacle_id,
                                                           obstacle_type=ObstacleType('pedestrian'),
                                                           obstacle_shape=self.shape,
                                                           initial_state=initial_state,
                                                           prediction=trajectory_prediction)

        self.commonroad_phantom_obstacle = PhantomObstacle(obstacle_id=self.obstacle_id,
                                                           prediction=set_based_prediction)

    def set_velocity(self, v):
        self.v = v

    def calc_trajectory(self, dt=0.1, duration=None):
        """
        Computes the positions reached by an object in a given time,
        based on its starting point, orientation, speed, and time step.

        dt: time step
        duration: time duration

        return: numpy array with the positions reached by the object
        """
        # load variables
        v = self.v
        orientation = self.orientation
        start_point = self.pos

        # calculate duration
        if duration is None:
            duration = self.trajectory_length/v

        # Compute the x and y components of the velocity
        vx = round(v * np.cos(orientation), 3)
        vy = round(v * np.sin(orientation), 3)

        # Compute the number of time steps
        num_steps = int(duration / dt) + 1

        # Create a matrix of time steps and velocities
        t = np.arange(num_steps + 1)[:, np.newaxis] * dt
        v_matrix = np.array([vx, vy])

        # Compute the trajectory
        trajectory = start_point + t * v_matrix.T

        self.trajectory = trajectory

        if self.calc_ped_traj_polygons:
            self.calc_traj_polygons()

        if self.create_cr_obst:
            self._create_cr_obstacle()
            self._create_cr_collision_object()

    def calc_goal_position(self, sidewalk):
        # define length of linestring (only used to calculate final destination)
        length = 10

        # calculation final coordinates
        end_x = self.pos_point.x + length * np.cos(self.orientation)
        end_y = self.pos_point.y + length * np.sin(self.orientation)
        end_point = Point(end_x, end_y)

        # create linestring
        line = LineString([self.pos_point, end_point])

        # calc intersection between line and sidewalk (goal position)
        goal_position = line.intersection(sidewalk.interiors)[0]

        # use further point if geom type is MultiPoint
        if goal_position.geom_type == 'MultiPoint':

            # convert multipoint to list of np.arrays
            points = [np.array(p.coords).flatten() for p in goal_position.geoms]

            # calc distances between points and ego pos
            distances = np.linalg.norm(np.stack(points) - self.pos, axis=1)

            # find index of max distance
            max_distance_index = np.argmax(distances)

            # assign point to object
            self.goal_position = points[max_distance_index]
        else:
            self.goal_position = np.array(goal_position.coords).flatten()

        # calculate length of trajectory
        self.trajectory_length = np.linalg.norm(self.goal_position - self.pos)

    def calc_traj_polygons(self):
        # create ped orientation array in order to use compute_vehicle_polygons function
        ped_orientation_np = np.ones(len(self.trajectory)) * self.orientation
        self.orientations = ped_orientation_np

        # calculate pedestrian trajectory polygons for each timestep
        self.traj_polygons = ohf.compute_vehicle_polygons(self.trajectory[:, 0], self.trajectory[:, 1],
                                                          ped_orientation_np, self.width, self.length)




