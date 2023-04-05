
from commonroad.scenario.obstacle import DynamicObstacle

import commonroad_dc.pycrcc as pycrcc

from commonroad_rp.reactive_planner import ReactivePlanner


def check_collision(planner: ReactivePlanner, ego_vehicle: DynamicObstacle):
    """Replaces ReactivePlanner.check_collision().

    The modifications allow checking for collisions
    after synchronization of the agents, avoiding inconsistent
    views on the scenario. 

    :param planner: The planner used by the agent.
    :param ego_vehicle: The ego obstacle.
    """

    ego = pycrcc.TimeVariantCollisionObject((planner.x_0.time_step+1) * planner._factor)
    ego.append_obstacle(pycrcc.RectOBB(0.5 * planner.vehicle_params.length, 0.5 * planner.vehicle_params.width,
                                        ego_vehicle.initial_state.orientation,
                                        ego_vehicle.initial_state.position[0], ego_vehicle.initial_state.position[1]))

    if not planner.collision_checker.collide(ego):
        return False
    else:
        try:
            goal_position = []

            if planner.goal_checker.goal.state_list[0].has_value("position"):
                for x in planner.reference_path:
                    if planner.goal_checker.goal.state_list[0].position.contains_point(x):
                        goal_position.append(x)
                s_goal_1, d_goal_1 = planner._co.convert_to_curvilinear_coords(goal_position[0][0], goal_position[0][1])
                s_goal_2, d_goal_2 = planner._co.convert_to_curvilinear_coords(goal_position[-1][0], goal_position[-1][1])
                s_goal = min(s_goal_1, s_goal_2)
                s_start, d_start = planner._co.convert_to_curvilinear_coords(
                    planner.planning_problem.initial_state.position[0],
                    planner.planning_problem.initial_state.position[1])
                s_current, d_current = planner._co.convert_to_curvilinear_coords(planner.x_0.position[0], planner.x_0.position[1])
                progress = ((s_current - s_start) / (s_goal - s_start))
            elif "time_step" in planner.goal_checker.goal.state_list[0].attributes:
                progress = ((planner.x_0.time_step+1) / planner.goal_checker.goal.state_list[0].time_step.end)
            else:
                print('Could not calculate progress')
                progress = None
        except:
            progress = None
            print('Could not calculate progress')

        collision_obj = planner.collision_checker.find_all_colliding_objects(ego)[0]
        if isinstance(collision_obj, pycrcc.TimeVariantCollisionObject):
            obj = collision_obj.obstacle_at_time((planner.x_0.time_step+1))
            center = obj.center()
            last_center = collision_obj.obstacle_at_time(planner.x_0.time_step).center()
            r_x = obj.r_x()
            r_y = obj.r_y()
            orientation = obj.orientation()
            planner.logger.log_collision(True, planner.vehicle_params.length, planner.vehicle_params.width, progress, center,
                                      last_center, r_x, r_y, orientation)
        else:
            planner.logger.log_collision(False, planner.vehicle_params.length, planner.vehicle_params.width, progress)
        return True
