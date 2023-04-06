import os
from typing import List

import matplotlib
from matplotlib import pyplot as plt
import numpy as np

from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.obstacle import DynamicObstacle

import commonroad_dc.pycrcc as pycrcc
from commonroad.scenario.scenario import Scenario
from commonroad.visualization.draw_params import MPDrawParams, DynamicObstacleParams
from commonroad.visualization.mp_renderer import MPRenderer

from commonroad_rp.configuration import Configuration
from commonroad_rp.reactive_planner import ReactivePlanner
from commonroad_rp.trajectories import TrajectorySample

from commonroad_rp.utility.visualization import draw_uncertain_predictions_lb, draw_uncertain_predictions_wale


# Color palette for plotting
colors =      ["#e3af22", "#d6e322", "#96e322", "#55e322", "#22e32f",
               "#22e36f", "#22e3af", "#22d6e3", "#2296e3", "#2255e3",
               "#2f22e3", "#6f22e3", "#af22e3", "#e322d6", "#e32296"]
darkcolors =  ["#9c0d00", "#8f9c00", "#5b9c00", "#279c00", "#009c0d",
               "#009c41", "#009c75", "#008f9c", "#005b9c", "#00279c",
               "#0d009c", "#41009c", "#75009c", "#9c008f", "#9c005b"]
lightcolors = ["#ffd569", "#f8ff69", "#c6ff69", "#94ff69", "#69ff70",
               "#69ffa3", "#69ffd5", "#69f8ff", "#69c6ff", "#6993ff",
               "#7069ff", "#a369ff", "#d569ff", "#ff69f8", "#ff69c5"]


def check_collision(planner: ReactivePlanner, ego_vehicle: DynamicObstacle, timestep: int):
    """Replaces ReactivePlanner.check_collision().

    The modifications allow checking for collisions
    after synchronization of the agents, avoiding inconsistent
    views on the scenario. 

    :param planner: The planner used by the agent.
    :param ego_vehicle: The ego obstacle.
    """

    ego = pycrcc.TimeVariantCollisionObject(timestep)
    ego.append_obstacle(pycrcc.RectOBB(0.5 * planner.vehicle_params.length, 0.5 * planner.vehicle_params.width,
                                       ego_vehicle.state_at_time(timestep).orientation,
                                       ego_vehicle.state_at_time(timestep).position[0],
                                       ego_vehicle.state_at_time(timestep).position[1]))

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
                s_current, d_current = planner._co.convert_to_curvilinear_coords(
                    ego_vehicle.state_at_time(timestep-1).position[0],
                    ego_vehicle.state_at_time(timestep-1).position[1])
                progress = ((s_current - s_start) / (s_goal - s_start))
            elif "time_step" in planner.goal_checker.goal.state_list[0].attributes:
                progress = (timestep-1 / planner.goal_checker.goal.state_list[0].time_step.end)
            else:
                print('Could not calculate progress')
                progress = None
        except:
            progress = None
            print('Could not calculate progress')

        collision_obj = planner.collision_checker.find_all_colliding_objects(ego)[0]
        if isinstance(collision_obj, pycrcc.TimeVariantCollisionObject):
            obj = collision_obj.obstacle_at_time(timestep)
            center = obj.center()
            last_center = collision_obj.obstacle_at_time(timestep-1).center()
            r_x = obj.r_x()
            r_y = obj.r_y()
            orientation = obj.orientation()
            planner.logger.log_collision(True, planner.vehicle_params.length, planner.vehicle_params.width, progress, center,
                                      last_center, r_x, r_y, orientation)
        else:
            planner.logger.log_collision(False, planner.vehicle_params.length, planner.vehicle_params.width, progress)
        return True


def visualize_multiagent_at_timestep(scenario: Scenario, planning_problem_list: List[PlanningProblem],
                                     agent_list: List[DynamicObstacle], timestep: int,
                                     config: Configuration, log_path: str,
                                     traj_set_list: List[List[TrajectorySample]] = None,
                                     ref_path_list: List[np.ndarray] = None,
                                     predictions: dict = None, visible_area = None,
                                     rnd: MPRenderer = None,
                                     plot_window: int = None):
    """
    Function to visualize planning result from the reactive planner for a given time step
    for all agents in a multiagent simulation.
    Replaces visualize_visualize_planner_at_timestep from visualization.py

    :param scenario: CommonRoad scenario object containing no dummy obstacles
    :param planning_problem_list: Planning problems of all agents
    :param agent_list: Dummy obstacles for all agents. Assumed to include the recorded path in the trajectory
    :param timestep: Time step of the scenario to plot
    :param config: Configuration object for plot/save settings
    :param log_path: Path to save the plot to (optional, depending on the config)
    :param traj_set_list: List of lists of sampled trajectories for each agent (optional)
    :param ref_path_list: Reference paths for every planner as polyline [(nx2) np.ndarray] (optional)
    :param predictions: Dictionary of all predictions (optional)
    :param visible_area: Visible sensor area (optional, ignored if more than one agent is plotted)
    :param rnd: MPRenderer object (optional: if none is passed, the function creates a new renderer object;
                otherwise it will visualize on the existing object)
    :param plot_window: Size of the margin of the plot, or minimum distance between the
                        plot border and all agents (optional)
    """

    # create renderer object (if no existing renderer is passed)
    if rnd is None:
        rnd = MPRenderer(figsize=(20, 10))

    if plot_window is not None:
        # focus on window around all agents
        left = None
        right = None
        top = None
        bottom = None
        for agent in agent_list:
            if left is None or agent.state_at_time(timestep).position[0] < left:
                left = agent.state_at_time(timestep).position[0]
            if right is None or agent.state_at_time(timestep).position[0] > right:
                right = agent.state_at_time(timestep).position[0]

            if bottom is None or agent.state_at_time(timestep).position[1] < bottom:
                bottom = agent.state_at_time(timestep).position[1]
            if top is None or agent.state_at_time(timestep).position[1] > top:
                top = agent.state_at_time(timestep).position[1]

        rnd.plot_limits = [-plot_window + left,
                           plot_window + right,
                           -plot_window + bottom,
                           plot_window + top]

    # Set obstacle parameters
    obs_params = MPDrawParams()
    obs_params.dynamic_obstacle.time_begin = timestep
    obs_params.dynamic_obstacle.draw_icon = config.debug.draw_icons
    obs_params.dynamic_obstacle.show_label = True
    obs_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#E37222"
    obs_params.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = "#003359"

    obs_params.static_obstacle.show_label = True
    obs_params.static_obstacle.occupancy.shape.facecolor = "#a30000"
    obs_params.static_obstacle.occupancy.shape.edgecolor = "#756f61"

    # visualize scenario
    scenario.draw(rnd, draw_params=obs_params)

    # Visualize agents and planning problems
    for i in range(len(agent_list)):

        # set ego vehicle draw params
        ego_params = DynamicObstacleParams()
        ego_params.time_begin = timestep
        ego_params.draw_icon = config.debug.draw_icons
        ego_params.show_label = True
        ego_params.vehicle_shape.occupancy.shape.facecolor = \
            colors[agent_list[i].obstacle_id % len(colors)]
        ego_params.vehicle_shape.occupancy.shape.edgecolor = \
            darkcolors[agent_list[i].obstacle_id % len(darkcolors)]
        ego_params.vehicle_shape.occupancy.shape.zorder = 50
        ego_params.vehicle_shape.occupancy.shape.opacity = 1

        # Visualize planning problem and agent
        planning_problem_list[i].draw(rnd)
        agent_list[i].draw(rnd, draw_params=ego_params)

    rnd.render()

    # draw visible sensor area
    if visible_area is not None and len(agent_list) == 1:
        if visible_area.geom_type == "MultiPolygon":
            for geom in visible_area.geoms:
                rnd.ax.fill(*geom.exterior.xy, "g", alpha=0.2, zorder=10)
        elif visible_area.geom_type == "Polygon":
            rnd.ax.fill(*visible_area.exterior.xy, "g", alpha=0.2, zorder=10)
        else:
            for obj in visible_area.geoms:
                if obj.geom_type == "Polygon":
                    rnd.ax.fill(*obj.exterior.xy, "g", alpha=0.2, zorder=10)

    # Visualize trajectories and paths
    for i in range(len(agent_list)):

        # visualize optimal trajectory
        pos = np.asarray([state.position for state in agent_list[i].prediction.trajectory.state_list[timestep:]])
        rnd.ax.plot(pos[:, 0], pos[:, 1], color=darkcolors[agent_list[i].obstacle_id % len(darkcolors)],
                    marker='x', markersize=1.5, zorder=21, linewidth=2, label='optimal trajectory')

        # visualize sampled trajectory bundle
        step = 1  # draw every trajectory (step=2 would draw every second trajectory)
        if traj_set_list is not None:
            for j in range(0, len(traj_set_list[i]), step):
                plt.plot(traj_set_list[i][j].cartesian.x[:traj_set_list[i][j]._actual_traj_length],
                         traj_set_list[i][j].cartesian.y[:traj_set_list[i][j]._actual_traj_length],
                         color=lightcolors[agent_list[i].obstacle_id % len(lightcolors)], zorder=20,
                         linewidth=0.2, alpha=1.0)

        # visualize reference path
        if ref_path_list is not None:
            rnd.ax.plot(ref_path_list[i][:, 0], ref_path_list[i][:, 1],
                        color=colors[agent_list[i].obstacle_id % len(colors)],
                        marker='.', markersize=1, zorder=19, linewidth=0.8, label='reference path')

    # visualize predictions
    if predictions is not None:
        if config.prediction.mode == "lanebased":
            draw_uncertain_predictions_lb(predictions, rnd.ax)
        elif config.prediction.mode == "walenet":
            draw_uncertain_predictions_wale(predictions, rnd.ax)

    # save as .png file
    if agent_list[0].obstacle_id in config.multiagent.save_individual_plots or config.debug.save_plots:
        plot_dir = os.path.join(log_path, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        if config.debug.gif:
            plt.savefig(f"{plot_dir}/{scenario.scenario_id}_{timestep}.png", format='png', dpi=300)
        else:
            plt.savefig(f"{plot_dir}/{scenario.scenario_id}_{timestep}.svg", format='svg')

    # show plot
    if agent_list[0].obstacle_id in config.multiagent.show_individual_plots or config.debug.show_plots:
        matplotlib.use("TkAgg")
        plt.pause(0.0001)
