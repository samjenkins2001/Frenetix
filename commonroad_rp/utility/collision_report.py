from commonroad_dc.boundary import boundary
import numpy as np
from commonroad_rp.utility import helper_functions as hf
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import (
    create_collision_checker,
    create_collision_object,
)
from commonroad_helper_functions.exceptions import NoLocalTrajectoryFoundError
from risk_assessment.utils.logistic_regression_symmetrical import get_protected_inj_prob_log_reg_ignore_angle
from commonroad.scenario.obstacle import (
    ObstacleRole,
    ObstacleType,
)
from risk_assessment.helpers.collision_helper_function import angle_range
from risk_assessment.harm_estimation import harm_model
from risk_assessment.visualization.collision_visualization import (
    collision_vis,
)


def coll_report(ego_vehicle, planner, scenario, planning_problem, collision_report_path):
    # check if the current state is collision-free
    vel_list = []
    # get ego position and orientation
    try:
        ego_pos = ego_vehicle[-1].initial_state.position
        # ego_pos = (
        #     self.scenario.obstacle_by_id(obstacle_id=agent.agent_id)
        #     .occupancy_at_time(time_step=time_step)
        #     .shape.center
        # )
        # print(ego_pos)

    except AttributeError:
        print("None-type error")
    (
        _,
        road_boundary,
    ) = boundary.create_road_boundary_obstacle(
        scenario=scenario,
        method="aligned_triangulation",
        axis=2,
    )

    if ego_vehicle[-1].initial_state.time_step == 0:
        ego_vel = ego_vehicle[-1].initial_state.velocity
        ego_yaw = ego_vehicle[-1].initial_state.orientation

        vel_list.append(ego_vel)
    else:
        ego_pos_last = ego_vehicle[-2].initial_state.position

        delta_ego_pos = ego_pos - ego_pos_last

        ego_vel = np.linalg.norm(delta_ego_pos) / scenario.dt

        vel_list.append(ego_vel)

        ego_yaw = np.arctan2(delta_ego_pos[1], delta_ego_pos[0])

    current_state_collision_object = hf.create_tvobstacle(
        traj_list=[
            [
                ego_pos[0],
                ego_pos[1],
                ego_yaw,
            ]
        ],
        box_length= planner.vehicle_params.length / 2,
        box_width=planner.vehicle_params.width / 2,
        start_time_step=ego_vehicle[-1].initial_state.time_step,
    )

    # Add road boundary to collision checker
    planner._cc.add_collision_object(road_boundary)

    if not planner._cc.collide(current_state_collision_object):
        return

    # get the colliding obstacle
    obs_id = None
    for obs in scenario.obstacles:
        co = create_collision_object(obs)
        if current_state_collision_object.collide(co):
            if obs.obstacle_id != ego_vehicle[-1].obstacle_id:
                if obs_id is None:
                    obs_id = obs.obstacle_id
                else:
                    print("More than one collision detected")
                    return

    # Collisoin with boundary
    if obs_id is None:
        ego_harm = get_protected_inj_prob_log_reg_ignore_angle(
            velocity=ego_vel, coeff=planner.params_harm
        )
        total_harm = ego_harm

        print("Collision with road boundary. (Harm: {:.2f})".format(ego_harm))
        return

    # get information of colliding obstace
    obs_pos = (
        scenario.obstacle_by_id(obstacle_id=obs_id)
        .occupancy_at_time(time_step=ego_vehicle[-1].initial_state.time_step)
        .shape.center
    )
    obs_pos_last = (
        scenario.obstacle_by_id(obstacle_id=obs_id)
        .occupancy_at_time(time_step=ego_vehicle[-1].initial_state.time_step - 1)
        .shape.center
    )
    obs_size = (
            scenario.obstacle_by_id(obstacle_id=obs_id).obstacle_shape.length
            * scenario.obstacle_by_id(obstacle_id=obs_id).obstacle_shape.width
    )

    # filter initial collisions
    if ego_vehicle[-1].initial_state.time_step < 1:
        print("Collision at initial state")
        return
    if (
            scenario.obstacle_by_id(obstacle_id=obs_id).obstacle_role
            == ObstacleRole.ENVIRONMENT
    ):
        obs_vel = 0
        obs_yaw = 0
    else:
        pos_delta = obs_pos - obs_pos_last

        obs_vel = np.linalg.norm(pos_delta) / scenario.dt
        if (
                scenario.obstacle_by_id(obstacle_id=obs_id).obstacle_role
                == ObstacleRole.DYNAMIC
        ):
            obs_yaw = np.arctan2(pos_delta[1], pos_delta[0])
        else:
            obs_yaw = scenario.obstacle_by_id(
                obstacle_id=obs_id
            ).initial_state.orientation

    # calculate crash angle
    pdof = angle_range(obs_yaw - ego_yaw + np.pi)
    rel_angle = np.arctan2(
        obs_pos_last[1] - ego_pos_last[1], obs_pos_last[0] - ego_pos_last[0]
    )
    ego_angle = angle_range(rel_angle - ego_yaw)
    obs_angle = angle_range(np.pi + rel_angle - obs_yaw)

    # calculate harm
    ego_harm, obs_harm, ego_obj, obs_obj = harm_model(
        scenario=scenario,
        ego_vehicle_sc=ego_vehicle,
        vehicle_params=planner.vehicle_params,
        ego_velocity=ego_vel,
        ego_yaw=ego_yaw,
        obstacle_id=obs_id,
        obstacle_size=obs_size,
        obstacle_velocity=obs_vel,
        obstacle_yaw=obs_yaw,
        pdof=pdof,
        ego_angle=ego_angle,
        obs_angle=obs_angle,
        modes=planner.params_risk,
        coeffs=planner.params_harm,
    )

    # if collision report should be shown
    collision_vis(
        scenario=scenario,
        ego_vehicle=ego_vehicle,
        destination=collision_report_path,
        ego_harm=ego_harm,
        ego_type=ego_obj.type,
        ego_v=ego_vel,
        ego_mass=ego_obj.mass,
        obs_harm=obs_harm,
        obs_type=obs_obj.type,
        obs_v=obs_vel,
        obs_mass=obs_obj.mass,
        pdof=pdof,
        ego_angle=ego_angle,
        obs_angle=obs_angle,
        time_step=ego_vehicle[-1].initial_state.time_step,
        modes=planner.params_risk,
        marked_vehicle=ego_vehicle[-1].obstacle_id,
        planning_problem=planning_problem,
        global_path=None,
        driven_traj=None,
    )
