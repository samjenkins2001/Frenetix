import os
from typing import List, Union, Dict

import matplotlib
from matplotlib import pyplot as plt
import imageio.v3 as iio
import numpy as np

from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType

import commonroad_dc.pycrcc as pycrcc
from commonroad.scenario.scenario import Scenario
from commonroad.visualization.draw_params import MPDrawParams, DynamicObstacleParams
from commonroad.visualization.mp_renderer import MPRenderer

from behavior_planner.FSM_model import State
from commonroad_rp.configuration import Configuration
from commonroad_rp.reactive_planner import ReactivePlanner
from commonroad_rp.trajectories import TrajectorySample

from commonroad_rp.utility.visualization import draw_uncertain_predictions_lb, draw_uncertain_predictions_wale

# Color palette for plotting
colors = ["#e3af22", "#d6e322", "#96e322", "#55e322", "#22e32f",
          "#22e36f", "#22e3af", "#22d6e3", "#2296e3", "#2255e3",
          "#2f22e3", "#6f22e3", "#af22e3", "#e322d6", "#e32296"]
darkcolors = ["#9c0d00", "#8f9c00", "#5b9c00", "#279c00", "#009c0d",
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
                s_goal_2, d_goal_2 = planner._co.convert_to_curvilinear_coords(goal_position[-1][0],
                                                                               goal_position[-1][1])
                s_goal = min(s_goal_1, s_goal_2)
                s_start, d_start = planner._co.convert_to_curvilinear_coords(
                    planner.planning_problem.initial_state.position[0],
                    planner.planning_problem.initial_state.position[1])
                s_current, d_current = planner._co.convert_to_curvilinear_coords(
                    ego_vehicle.state_at_time(timestep - 1).position[0],
                    ego_vehicle.state_at_time(timestep - 1).position[1])
                progress = ((s_current - s_start) / (s_goal - s_start))
            elif "time_step" in planner.goal_checker.goal.state_list[0].attributes:
                progress = (timestep - 1 / planner.goal_checker.goal.state_list[0].time_step.end)
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
            last_center = collision_obj.obstacle_at_time(timestep - 1).center()
            r_x = obj.r_x()
            r_y = obj.r_y()
            orientation = obj.orientation()
            planner.logger.log_collision(True, planner.vehicle_params.length, planner.vehicle_params.width, progress,
                                         center,
                                         last_center, r_x, r_y, orientation)
        else:
            planner.logger.log_collision(False, planner.vehicle_params.length, planner.vehicle_params.width, progress)
        return True


def visualize_multiagent_at_timestep(scenario: Scenario, planning_problem_list: List[PlanningProblem],
                                     agent_list: List[DynamicObstacle], timestep: int,
                                     config: Configuration, log_path: str,
                                     traj_set_list: List[List[TrajectorySample]] = None,
                                     ref_path_list: List[np.ndarray] = None,
                                     predictions: dict = None, visible_area=None,
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
    if (len(agent_list) == 1 and agent_list[0].obstacle_id in config.multiagent.save_individual_plots) \
            or config.debug.save_plots:
        plot_dir = os.path.join(log_path, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        if (len(agent_list) == 1 and agent_list[0].obstacle_id in config.multiagent.save_individual_gifs) \
                or config.debug.gif:
            plt.savefig(f"{plot_dir}/{scenario.scenario_id}_{timestep}.png", format='png', dpi=300)
        else:
            plt.savefig(f"{plot_dir}/{scenario.scenario_id}_{timestep}.svg", format='svg')

    # show plot
    if agent_list[0].obstacle_id in config.multiagent.show_individual_plots or config.debug.show_plots:
        matplotlib.use("TkAgg")
        plt.pause(0.0001)


def make_gif(scenario: Scenario, time_steps: Union[range, List[int]],
             log_path: str, duration: float = 0.1):
    """
    Function to create an animated GIF from single images of planning results at each time step.
    Replaces commonroad_rp.utility.visualization.make_gif

    Images are assumed to be saved as <log_path>/plots/<scenario_id>_<timestep>.png
    In contrast to the original function, this one does not check the configuration
    in order to simplify independent configurations on agent and simulation level.

    :param scenario: CommonRoad scenario object.
    :param time_steps: List or range of time steps to include in the GIF
    :param log_path: Base path containing the plots folder with the input images
    :param duration: Duration of the individual frames (default: 0.1s)
    """

    print("...Generating GIF")
    images = []
    filenames = []

    # directory, where single images are outputted (see visualize_planner_at_timestep())
    path_images = os.path.join(log_path, "plots")

    for step in time_steps:
        im_path = os.path.join(path_images, str(scenario.scenario_id) + "_{}.png".format(step))
        filenames.append(im_path)

    for filename in filenames:
        images.append(iio.imread(filename))

    iio.imwrite(os.path.join(log_path, str(scenario.scenario_id) + ".gif"),
                images, duration=duration)


def collision_vis(scenario: Scenario, ego_vehicle: DynamicObstacle, destination: str,
                  ego_harm: float, ego_type: ObstacleType, ego_v: float, ego_mass: float,
                  obs_harm: float, obs_type: ObstacleType, obs_v: float, obs_mass: float,
                  pdof: float, ego_angle: float, obs_angle: float,
                  time_step: int, modes: Dict, planning_problem: PlanningProblem = None,
                  global_path: np.ndarray = None, driven_traj: List[State] = None):
    """
    Create a report for visualization of the collision and respective harm.

    Creates a collision report and saves the file in the destination folder
    in a subdirectory called "collisions".
    Replaces risk_assessment.visualization.collision_visualization.collision_vis
    Allows using the full ego obstacle.

    :param scenario: Considered Scenario.
    :param ego_vehicle: Ego obstacle, may contain both past and future trajectories.
    :param destination: Path to save output to.
    :param ego_type: Type of the ego vehicle (usually CAR).
    :param ego_harm: Harm for the ego vehicle.
    :param ego_v: Impact speed of the ego vehicle.
    :param ego_mass: Mass of the ego vehicle.
    :param obs_harm: Harm for the obstacle.
    :param obs_type: Type of obstacle.
    :param obs_v: Velocity of the obstacle.
    :param obs_mass: Estimated mass of the obstacle.
    :param pdof: Principle degree of force.
    :param ego_angle: Angle of impact area for the ego vehicle.
    :param obs_angle: Angle of impact area for the obstacle.
    :param time_step: Current time step.
    :param modes: Risk modes. Read from risk.json.
    :param planning_problem: Considered planning problem. Defaults to None.
    :param global_path: Global path for the planning problem. Defaults to None.
    :param driven_traj: Already driven trajectory of the ego vehicle. Defaults to None.
    """

    # clear everything
    plt.cla()

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    fig.set_size_inches(5.833, 8.25)

    ego_params = DynamicObstacleParams()
    ego_params.time_begin = time_step
    ego_params.draw_icon = True
    ego_params.vehicle_shape.occupancy.shape.facecolor = "#E37222"
    ego_params.vehicle_shape.occupancy.shape.edgecolor = "#9C4100"
    ego_params.vehicle_shape.occupancy.shape.zorder = 50
    ego_params.vehicle_shape.occupancy.shape.opacity = 1

    # set plot limits to show the road section around the ego vehicle
    position = ego_vehicle.state_at_time(time_step).position
    plot_limits = [position[0] - 20,
                   position[0] + 20,
                   position[1] - 20,
                   position[1] + 20]

    scenario_params = MPDrawParams()
    scenario_params.time_begin = time_step
    scenario_params.dynamic_obstacle.show_label = True
    scenario_params.dynamic_obstacle.draw_icon = True
    scenario_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#0065BD"
    scenario_params.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = "#003359"
    scenario_params.dynamic_obstacle.vehicle_shape.occupancy.shape.zorder = 50
    scenario_params.dynamic_obstacle.vehicle_shape.occupancy.shape.opacity = 1

    rnd = MPRenderer(ax=ax2, plot_limits=plot_limits)
    # plot the scenario at the current time step
    scenario.draw(rnd, draw_params=scenario_params)

    plt.gca().set_aspect('equal')

    # draw the planning problem
    if planning_problem is not None:
        planning_problem.draw(rnd)

    # mark the ego vehicle
    if ego_vehicle is not None:
        ego_vehicle.draw(rnd, draw_params=ego_params)

    rnd.render()

    # Draw global path
    if global_path is not None:
        plt.plot(global_path[:, 0], global_path[:, 1], color='blue',
                 zorder=20, label='global path')

    # draw driven trajectory
    if driven_traj is not None:
        x = [state.position[0] for state in driven_traj]
        y = [state.position[1] for state in driven_traj]
        plt.plot(x, y, color='green', zorder=25)

    # get the target time to show it in the title
    if hasattr(planning_problem.goal.state_list[0], 'time_step'):
        target_time_string = ('Target-time: %.1f s - %.1f s' %
                              (planning_problem.goal.state_list[0].
                               time_step.start * scenario.dt,
                               planning_problem.goal.state_list[0].
                               time_step.end * scenario.dt))
    else:
        target_time_string = 'No target-time'

    plt.legend()
    plt.title('Time: {0:.1f} s'.format(time_step * scenario.dt) + '    ' +
              target_time_string)

    # get mode description
    if modes["harm_mode"] == "log_reg":
        mode = "logistic regression"
        harm = "P(MAIS 3+)"
    elif modes["harm_mode"] == "ref_speed":
        mode = "reference speed"
        harm = "P(MAIS 3+)"
    elif modes["harm_mode"] == "gidas":
        mode = "GIDAS P(MAIS 2+)"
        harm = "P(MAIS 2+)"
    else:
        mode = "None"

    # get angle mode
    if modes["ignore_angle"] is True or modes["harm_mode"] == "gidas":
        angle = "ignoring impact areas"
    else:
        if modes["reduced_angle_areas"]:
            angle = "considering impact areas reduced on front, side, and " \
                    "rear crashes"
        else:
            angle = "considering impact areas according to the clock system"

        if modes["sym_angle"]:
            angle += " with symmetric coefficients"
        else:
            angle += " with asymmetric coefficients"

    # description of crash
    description = "Collision at {:.1f} s in ". \
                      format(time_step * scenario.dt) + \
                  str(scenario.scenario_id) + \
                  "\n\nCalculate harm using the " + mode + " model by " + angle + \
                  "\n\nEgo vehicle harm " + harm + ": {:.3f}".format(ego_harm) + \
                  "\nObstacle harm " + harm + ": {:.3f}".format(obs_harm) + \
                  "\n\nCrash parameters:\n\nEgo type: " + str(ego_type)[13:] + \
                  "\nEgo velocity: {:.2f}m/s". \
                      format(ego_v) + \
                  "\nEgo mass: {:.0f}kg".format(ego_mass) + \
                  "\nImpact angle for the ego vehicle: {:.2f}°". \
                      format(ego_angle * 180 / np.pi) + \
                  "\n\nObstacle type: " + str(obs_type)[13:]
    if obs_mass is not None and obs_v is not None and obs_angle is not None:
        description += "\nObstacle velocity: {:.2f}m/s".format(obs_v) + \
                       "\nObstacle mass: {:.0f}kg".format(obs_mass) + \
                       "\nImpact angle for the obstacle: {:.2f}°". \
                           format(obs_angle * 180 / np.pi)
    description += "\n\nPrinciple degree of force: {:.2f}°". \
        format(pdof * 180 / np.pi)

    # add description of crash
    ax1.axis('off')
    ax1.text(0, 1, description, verticalalignment='top', fontsize=8,
             wrap=True)

    fig.suptitle("Collision in " + str(scenario.scenario_id) + " detected",
                 fontsize=16)

    # Create directory for pictures
    destination = os.path.join(destination, "collisions")
    if not os.path.exists(destination):
        os.makedirs(destination)

    plt.savefig(destination + "/" + str(scenario.scenario_id) + ".svg", format="svg")
    plt.close()
