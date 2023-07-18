import numpy as np
import math
from typing import List
import logging
from commonroad_rp.prediction_helpers import collision_checker_prediction
from risk_assessment.utils.logistic_regression_symmetrical import get_protected_inj_prob_log_reg_ignore_angle

from commonroad_dc.collision.trajectory_queries.trajectory_queries import trajectories_collision_static_obstacles
from commonroad_dc.collision.trajectory_queries.trajectory_queries import trajectory_preprocess_obb_sum
import commonroad_dc.pycrcc as pycrcc

from commonroad.scenario.trajectory import Trajectory

from commonroad_rp.utility import helper_functions as hf
from commonroad_rp.utility.utils_coordinate_system import interpolate_angle
from commonroad_rp.trajectories import TrajectorySample, CartesianSample, CurviLinearSample

# precision value
_EPS = 1e-5

# get logger
msg_logger = logging.getLogger("Message_logger")


def check_feasibility(self, trajectories: List[TrajectorySample], queue_1=None, queue_2=None, queue_3=None):
    """
    Checks the kinematics of given trajectories in a bundle and computes the cartesian trajectory information
    Lazy evaluation, only kinematically feasible trajectories are evaluated further

    :param trajectories: The list of trajectory samples to check
    :param queue_1: Multiprocessing.Queue() object for storing feasible trajectories
    :param queue_2: Multiprocessing.Queue() object for storing infeasible trajectories (only vor visualization)
    :param queue_3: Multiprocessing.Queue() object for storing reason for infeasible trajectory in list
    :return: The list of output trajectories
    """
    # initialize lists for output trajectories
    # infeasible trajectory list is only used for visualization when self._draw_traj_set is True
    infeasible_count_kinematics = np.zeros(10)
    feasible_trajectories = list()
    infeasible_trajectories = list()

    # loop over list of trajectories
    for trajectory in trajectories:
        # create time array and precompute time interval information
        t = np.arange(0, np.round(trajectory.trajectory_long.delta_tau + trajectory.dt, 5), trajectory.dt)
        t2 = np.square(t)
        t3 = t2 * t
        t4 = np.square(t2)
        t5 = t4 * t

        # length of the trajectory sample (i.e., number of time steps. can be smaller than planning horizon)
        traj_len = len(t)

        # initialize long. (s) and lat. (d) state vectors
        s = np.zeros(traj_len)
        s_velocity = np.zeros(traj_len)
        s_acceleration = np.zeros(traj_len)
        d = np.zeros(traj_len)
        d_velocity = np.zeros(traj_len)
        d_acceleration = np.zeros(traj_len)

        # compute longitudinal position, velocity, acceleration from trajectory sample
        s[:traj_len] = trajectory.trajectory_long.calc_position(t, t2, t3, t4, t5)  # lon pos
        s_velocity[:traj_len] = trajectory.trajectory_long.calc_velocity(t, t2, t3, t4)  # lon velocity
        s_acceleration[:traj_len] = trajectory.trajectory_long.calc_acceleration(t, t2, t3)  # lon acceleration

        # At low speeds, we have to sample the lateral motion over the travelled distance rather than time.
        if not self._LOW_VEL_MODE:
            d[:traj_len] = trajectory.trajectory_lat.calc_position(t, t2, t3, t4, t5)  # lat pos
            d_velocity[:traj_len] = trajectory.trajectory_lat.calc_velocity(t, t2, t3, t4)  # lat velocity
            d_acceleration[:traj_len] = trajectory.trajectory_lat.calc_acceleration(t, t2, t3)  # lat acceleration
        else:
            # compute normalized travelled distance for low velocity mode of lateral planning
            s1 = s[:traj_len] - s[0]
            s2 = np.square(s1)
            s3 = s2 * s1
            s4 = np.square(s2)
            s5 = s4 * s1

            # compute lateral position, velocity, acceleration from trajectory sample
            d[:traj_len] = trajectory.trajectory_lat.calc_position(s1, s2, s3, s4, s5)  # lat pos
            # in LOW_VEL_MODE d_velocity is actually d' (see Diss. Moritz Werling  p.124)
            d_velocity[:traj_len] = trajectory.trajectory_lat.calc_velocity(s1, s2, s3, s4)  # lat velocity
            d_acceleration[:traj_len] = trajectory.trajectory_lat.calc_acceleration(s1, s2, s3)  # lat acceleration

        # precision for near zero velocities from evaluation of polynomial coefficients
        # set small velocities to zero
        s_velocity[np.abs(s_velocity) < _EPS] = 0.0
        d_velocity[np.abs(d_velocity) < _EPS] = 0.0

        # Initialize trajectory state vectors
        # (Global) Cartesian positions x, y
        x = np.zeros(traj_len)
        y = np.zeros(traj_len)
        # (Global) Cartesian velocity v and acceleration a
        v = np.zeros(traj_len)
        a = np.zeros(traj_len)
        # Orientation theta: Cartesian (gl) and Curvilinear (cl)
        theta_gl = np.zeros(traj_len)
        theta_cl = np.zeros(traj_len)
        # Curvature kappa : Cartesian (gl) and Curvilinear (cl)
        kappa_gl = np.zeros(traj_len)
        kappa_cl = np.zeros(traj_len)

        # Initialize Feasibility boolean
        feasible = True

        if not self._draw_traj_set:
            # pre-filter with quick underapproximative check for feasibility
            if np.any(np.abs(s_acceleration) > self.vehicle_params.a_max):
                msg_logger.debug(f"Acceleration {np.max(np.abs(s_acceleration))}")
                feasible = False
                infeasible_count_kinematics[1] += 1
                infeasible_trajectories.append(trajectory)
                continue
            if np.any(s_velocity < -_EPS):
                msg_logger.debug(f"Velocity {min(s_velocity)} at step")
                feasible = False
                infeasible_count_kinematics[2] += 1
                infeasible_trajectories.append(trajectory)
                continue

        infeasible_count_kinematics_traj = np.zeros(10)
        for i in range(0, traj_len):
            # compute orientations
            # see Appendix A.1 of Moritz Werling's PhD Thesis for equations
            if not self._LOW_VEL_MODE:
                if s_velocity[i] > 0.001:
                    dp = d_velocity[i] / s_velocity[i]
                else:
                    # if abs(d_velocity[i]) > 0.001:
                    #     dp = None
                    # else:
                    dp = 0.
                # see Eq. (A.8) from Moritz Werling's Diss
                ddot = d_acceleration[i] - dp * s_acceleration[i]

                if s_velocity[i] > 0.001:
                    dpp = ddot / (s_velocity[i] ** 2)
                else:
                    # if np.abs(ddot) > 0.00003:
                    #     dpp = None
                    # else:
                    dpp = 0.
            else:
                dp = d_velocity[i]
                dpp = d_acceleration[i]

            # factor for interpolation
            s_idx = np.argmax(self._co.ref_pos > s[i]) - 1
            if s_idx + 1 >= len(self._co.ref_pos):
                feasible = False
                infeasible_count_kinematics_traj[3] = 1
                break
            s_lambda = (s[i] - self._co.ref_pos[s_idx]) / (self._co.ref_pos[s_idx + 1] - self._co.ref_pos[s_idx])

            # compute curvilinear (theta_cl) and global Cartesian (theta_gl) orientation
            if s_velocity[i] > 0.001:
                # LOW VELOCITY MODE: dp = d_velocity[i]
                # HIGH VELOCITY MODE: dp = d_velocity[i]/s_velocity[i]
                theta_cl[i] = np.arctan2(dp, 1.0)

                theta_gl[i] = theta_cl[i] + interpolate_angle(
                    s[i],
                    self._co.ref_pos[s_idx],
                    self._co.ref_pos[s_idx + 1],
                    self._co.ref_theta[s_idx],
                    self._co.ref_theta[s_idx + 1])
            else:
                if self._LOW_VEL_MODE:
                    # dp = velocity w.r.t. to travelled arclength (s)
                    theta_cl[i] = np.arctan2(dp, 1.0)

                    theta_gl[i] = theta_cl[i] + interpolate_angle(
                        s[i],
                        self._co.ref_pos[s_idx],
                        self._co.ref_pos[s_idx + 1],
                        self._co.ref_theta[s_idx],
                        self._co.ref_theta[s_idx + 1])
                else:
                    # in stillstand (s_velocity~0) and High velocity mode: assume vehicle keeps global orientation
                    theta_gl[i] = self.x_0.orientation if i == 0 else theta_gl[i - 1]

                    theta_cl[i] = theta_gl[i] - interpolate_angle(
                        s[i],
                        self._co.ref_pos[s_idx],
                        self._co.ref_pos[s_idx + 1],
                        self._co.ref_theta[s_idx],
                        self._co.ref_theta[s_idx + 1])

            # Interpolate curvature of reference path k_r at current position
            k_r = (self._co.ref_curv[s_idx + 1] - self._co.ref_curv[s_idx]) * s_lambda + self._co.ref_curv[
                s_idx]
            # Interpolate curvature rate of reference path k_r_d at current position
            k_r_d = (self._co.ref_curv_d[s_idx + 1] - self._co.ref_curv_d[s_idx]) * s_lambda + \
                    self._co.ref_curv_d[s_idx]

            # compute global curvature (see appendix A of Moritz Werling's PhD thesis)
            oneKrD = (1 - k_r * d[i])
            cosTheta = math.cos(theta_cl[i])
            tanTheta = np.tan(theta_cl[i])
            kappa_gl[i] = (dpp + (k_r * dp + k_r_d * d[i]) * tanTheta) * cosTheta * (cosTheta / oneKrD) ** 2 + (
                    cosTheta / oneKrD) * k_r
            kappa_cl[i] = kappa_gl[i] - k_r

            # compute (global) Cartesian velocity
            v[i] = s_velocity[i] * (oneKrD / (math.cos(theta_cl[i])))

            # compute (global) Cartesian acceleration
            a[i] = s_acceleration[i] * oneKrD / cosTheta + ((s_velocity[i] ** 2) / cosTheta) * (
                    oneKrD * tanTheta * (kappa_gl[i] * oneKrD / cosTheta - k_r) - (
                    k_r_d * d[i] + k_r * dp))

            # CHECK KINEMATIC CONSTRAINTS (remove infeasible trajectories)
            # velocity constraint
            if v[i] < -_EPS:
                feasible = False
                infeasible_count_kinematics_traj[4] = 1
                if not self._draw_traj_set and not self._kinematic_debug:
                    break
            # curvature constraint
            kappa_max = np.tan(self.vehicle_params.delta_max) / self.vehicle_params.wheelbase
            if abs(kappa_gl[i]) > kappa_max:
                feasible = False
                infeasible_count_kinematics_traj[5] = 1
                if not self._draw_traj_set and not self._kinematic_debug:
                    break
            # yaw rate (orientation change) constraint
            yaw_rate = (theta_gl[i] - theta_gl[i - 1]) / self.dT if i > 0 else 0.
            theta_dot_max = kappa_max * v[i]
            if abs(round(yaw_rate, 5)) > theta_dot_max:
                feasible = False
                infeasible_count_kinematics_traj[6] = 1
                if not self._draw_traj_set and not self._kinematic_debug:
                    break
            # curvature rate constraint
            # TODO: chck if kappa_gl[i-1] ??
            steering_angle = np.arctan2(self.vehicle_params.wheelbase * kappa_gl[i], 1.0)
            kappa_dot_max = self.vehicle_params.v_delta_max / (self.vehicle_params.wheelbase *
                                                               math.cos(steering_angle) ** 2)
            kappa_dot = (kappa_gl[i] - kappa_gl[i - 1]) / self.dT if i > 0 else 0.
            if abs(kappa_dot) > kappa_dot_max:
                feasible = False
                infeasible_count_kinematics_traj[7] = 1
                if not self._draw_traj_set and not self._kinematic_debug:
                    break
            # acceleration constraint (considering switching velocity, see vehicle models documentation)
            v_switch = self.vehicle_params.v_switch
            a_max = self.vehicle_params.a_max * v_switch / v[i] if v[i] > v_switch else self.vehicle_params.a_max
            a_min = -self.vehicle_params.a_max
            if not a_min <= a[i] <= a_max:
                feasible = False
                infeasible_count_kinematics_traj[8] = 1
                if not self._draw_traj_set and not self._kinematic_debug:
                    break

        # if selected polynomial trajectory is feasible, store it's Cartesian and Curvilinear trajectory
        if feasible or self._draw_traj_set:
            # Extend Trajectory to get same lenth
            # t_ext = np.arange(1, len(s) - traj_len + 1, 1) * trajectory.dt
            # s[traj_len:] = s[traj_len-1] + t_ext * v[traj_len-1]
            # d[traj_len:] = d[traj_len-1]
            for i in range(0, len(s)):
                # compute (global) Cartesian position
                pos: np.ndarray = self._co.convert_to_cartesian_coords(s[i], d[i])
                if pos is not None:
                    x[i] = pos[0]
                    y[i] = pos[1]
                else:
                    feasible = False
                    infeasible_count_kinematics_traj[9] = 1
                    msg_logger.debug("Out of projection domain")
                    break

            if feasible or self._draw_traj_set:
                # store Cartesian trajectory
                trajectory.cartesian = CartesianSample(x, y, theta_gl, v, a, kappa_gl,
                                                       kappa_dot=np.append([0], np.diff(kappa_gl)),
                                                       current_time_step=traj_len)

                # store Curvilinear trajectory
                trajectory.curvilinear = CurviLinearSample(s, d, theta_cl,
                                                           ss=s_velocity, sss=s_acceleration,
                                                           dd=d_velocity, ddd=d_acceleration,
                                                           current_time_step=traj_len)

                trajectory.actual_traj_length = traj_len
                # check if trajectories planning horizon is shorter than expected and extend if necessary
                # shrt = trajectory.cartesian.current_time_step
                # if self.N + 1 > trajectory.cartesian.current_time_step:
                # trajectory = hf.shrink_trajectory(trajectory, shrt)
                # trajectory.enlarge(self.dT)
                # assert self.N + 1 == trajectory.cartesian.current_time_step == len(trajectory.cartesian.x) == \
                #       len(trajectory.cartesian.y) == len(trajectory.cartesian.theta), \
                #       '<ReactivePlanner/kinematics>:  Lenghts of state variables is not equal.'

            if feasible:
                feasible_trajectories.append(trajectory)
            elif not feasible and self._draw_traj_set:
                infeasible_trajectories.append(trajectory)

        infeasible_count_kinematics += infeasible_count_kinematics_traj

    if self._multiproc:
        # store feasible trajectories in Queue 1
        queue_1.put(feasible_trajectories)
        # if visualization is required: store infeasible trajectories in Queue 1
        if self._draw_traj_set:
            queue_2.put(infeasible_trajectories)
        if self._kinematic_debug:
            queue_3.put(infeasible_count_kinematics)
    else:
        return feasible_trajectories, infeasible_trajectories, infeasible_count_kinematics


def trajectory_collision_check(self, feasible_trajectories):
    """
    Checks valid trajectories for collisions with static obstacles
    :param feasible_trajectories: feasible trajectories list
    :return trajectory: optimal feasible trajectory or None
    """
    # go through sorted list of sorted trajectories and check for collisions
    for trajectory in feasible_trajectories:
        # Add Occupancy of Trajectory to do Collision Checks later
        cart_traj = self._compute_cart_traj(trajectory)
        occupancy = self.convert_state_list_to_commonroad_object(cart_traj.state_list)
        # get collision_object
        coll_obj = self.create_coll_object(occupancy, self.vehicle_params, self.x_0)

        # TODO: Check kinematic checks in cpp. no valid traj available
        if self.use_prediction:
            collision_detected = collision_checker_prediction(
                predictions=self.predictions,
                scenario=self.scenario,
                ego_co=coll_obj,
                frenet_traj=trajectory,
                ego_state=self.x_0,
            )
            if collision_detected:
                self._infeasible_count_collision += 1
        else:
            collision_detected = False

        leaving_road_at = trajectories_collision_static_obstacles(
            trajectories=[coll_obj],
            static_obstacles=self.road_boundary,
            method="grid",
            num_cells=32,
            auto_orientation=True,
        )
        if leaving_road_at[0] != -1:
            coll_time_step = leaving_road_at[0] - self.x_0.time_step
            coll_vel = trajectory.cartesian.v[coll_time_step]

            boundary_harm = get_protected_inj_prob_log_reg_ignore_angle(
                velocity=coll_vel, coeff=self.params_harm
            )

        else:
            boundary_harm = 0

        # Save Status of Trajectory to sort for alternative
        trajectory.boundary_harm = boundary_harm
        trajectory._coll_detected = collision_detected

        if not collision_detected and boundary_harm == 0:
            return trajectory

    return None


def create_coll_object(self, trajectory, vehicle_params, ego_state):
    """Create a collision_object of the trajectory for collision checking with road
    boundary and with other vehicles."""

    collision_object_raw = hf.create_tvobstacle_trajectory(
        traj_list=trajectory,
        box_length=vehicle_params.length / 2,
        box_width=vehicle_params.width / 2,
        start_time_step=ego_state.time_step,
    )
    # if the preprocessing fails, use the raw trajectory
    collision_object, err = trajectory_preprocess_obb_sum(
        collision_object_raw
    )
    if err:
        collision_object = collision_object_raw

    return collision_object


def check_goal_reached(self):
    # Get the ego vehicle
    self.goal_checker.register_current_state(self.x_0)
    self.goal_status, self.goal_message, self.full_goal_status = self.goal_checker.goal_reached_status()


def check_collision(self, ego_vehicle):

    ego = pycrcc.TimeVariantCollisionObject((self.x_0.time_step))
    ego.append_obstacle(pycrcc.RectOBB(0.5 * self.vehicle_params.length, 0.5 * self.vehicle_params.width,
                                       ego_vehicle.initial_state.orientation,
                                       ego_vehicle.initial_state.position[0], ego_vehicle.initial_state.position[1]))

    if not self.collision_checker.collide(ego):
        return False
    else:
        try:
            goal_position = []

            if self.goal_checker.goal.state_list[0].has_value("position"):
                for x in self.reference_path:
                    if self.goal_checker.goal.state_list[0].position.contains_point(x):
                        goal_position.append(x)
                s_goal_1, d_goal_1 = self._co.convert_to_curvilinear_coords(goal_position[0][0], goal_position[0][1])
                s_goal_2, d_goal_2 = self._co.convert_to_curvilinear_coords(goal_position[-2][0], goal_position[-2][1])
                s_goal = min(s_goal_1, s_goal_2)
                s_start, d_start = self._co.convert_to_curvilinear_coords(
                    self.planning_problem.initial_state.position[0],
                    self.planning_problem.initial_state.position[1])
                s_current, d_current = self._co.convert_to_curvilinear_coords(self.x_0.position[0], self.x_0.position[1])
                progress = ((s_current - s_start) / (s_goal - s_start))
            elif "time_step" in self.goal_checker.goal.state_list[0].attributes:
                progress = ((self.x_0.time_step -1) / self.goal_checker.goal.state_list[0].time_step.end)
            else:
                msg_logger.error('Could not calculate progress')
                progress = None
        except:
            progress = None
            msg_logger.error('Could not calculate progress')

        collision_obj = self.collision_checker.find_all_colliding_objects(ego)[0]
        if isinstance(collision_obj, pycrcc.TimeVariantCollisionObject):
            obj = collision_obj.obstacle_at_time((self.x_0.time_step -1))
            center = obj.center()
            last_center = collision_obj.obstacle_at_time(self.x_0.time_step-2).center()
            r_x = obj.r_x()
            r_y = obj.r_y()
            orientation = obj.orientation()
            self.logger.log_collision(True, self.vehicle_params.length, self.vehicle_params.width, progress, center,
                                      last_center, r_x, r_y, orientation)
        else:
            self.logger.log_collision(False, self.vehicle_params.length, self.vehicle_params.width, progress)
        return True


def shift_orientation(self, trajectory: Trajectory, interval_start=-np.pi, interval_end=np.pi):
    for state in trajectory.state_list:
        while state.orientation < interval_start:
            state.orientation += 2 * np.pi
        while state.orientation > interval_end:
            state.orientation -= 2 * np.pi
    return trajectory
