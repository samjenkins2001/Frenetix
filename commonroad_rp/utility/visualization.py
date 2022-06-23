__author__ = "Gerald Würsching"
__copyright__ = "TUM Cyber-Physical Systems Group"
__version__ = "0.5"
__maintainer__ = "Gerald Würsching"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Beta"


# standard imports
from typing import List, Tuple
import os

# third party
import matplotlib.pyplot as plt
import numpy as np

# commonroad-io
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.trajectory import State
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.geometry.shape import Rectangle
from prediction.utils.visualization import draw_uncertain_predictions
# commonroad_dc
from commonroad_dc import pycrcc

# commonroad-rp
from commonroad_rp.trajectories import TrajectorySample
from commonroad_rp.configuration import Configuration


def visualize_collision_checker(scenario: Scenario, cc: pycrcc.CollisionChecker):
    """
    Visualizes the collision checker, i.e., all collision objects and, if applicable, the road boundary.
    :param scenario CommonRoad scenario object
    :param cc pycrcc.CollisionChecker object
    """
    rnd = MPRenderer(figsize=(20, 10))
    scenario.lanelet_network.draw(rnd)
    cc.draw(rnd)
    rnd.render(show=True)


def visualize_planner_at_timestep(scenario: Scenario, planning_problem: PlanningProblem, ego: DynamicObstacle,
                                  timestep: int, config: Configuration, traj_set: List[TrajectorySample] = None,
                                  ref_path: np.ndarray = None, rnd: MPRenderer = None, predictions: dict = None):
    """
    Function to visualize planning result from the reactive planner for a given time step
    :param scenario: CommonRoad scenario object
    :param planning_problem CommonRoad Planning problem object
    :param ego: Ego vehicle as CommonRoad DynamicObstacle object
    :param pos: positions of planned trajectory [(nx2) np.ndarray]
    :param timestep: current time step of scenario to plot
    :param config: Configuration object for plot/save settings
    :param traj_set: List of sampled trajectories (optional)
    :param ref_path: Reference path for planner as polyline [(nx2) np.ndarray] (optional)
    :param rnd: MPRenderer object (optional: if none is passed, the function creates a new renderer object; otherwise it
    will visualize on the existing object)
    :param save_path: Path to save plot as .png (optional)
    """
    # create renderer object (if no existing renderer is passed)
    if rnd is None:
        rnd = MPRenderer(figsize=(20, 10))
    # visualize scenario
    scenario.draw(rnd, draw_params={'time_begin': timestep, 'dynamic_obstacle': {"draw_icon": config.debug.draw_icons}})
    # visualize planning problem
    planning_problem.draw(rnd, draw_params={'planning_problem': {'initial_state': {'state': {
                'draw_arrow': False, "radius": 0.5}}}})
    # visualize ego vehicle
    ego.draw(rnd, draw_params={"time_begin": timestep,
                               "dynamic_obstacle": {
                                   "vehicle_shape": {
                                       "occupancy": {
                                           "shape": {
                                               "rectangle": {
                                                   "facecolor": "#E37222",
                                                   "edgecolor": '#E37222',
                                                   "zorder": 50,
                                                   "opacity": 1
                                               }
                                           }
                                       }
                                   }
                               }
                               })
    # render scenario and ego vehicle
    rnd.render()

    # visualize optimal trajectory
    pos = np.asarray([state.position for state in ego.prediction.trajectory.state_list])
    rnd.ax.plot(pos[:, 0], pos[:, 1], color='k', marker='x', markersize=1.5, zorder=21, linewidth=1.5,
                label='optimal trajectory')

    # visualize sampled trajectory bundle
    step = 1  # draw every trajectory (step=2 would draw every second trajectory)
    if traj_set is not None:
        for i in range(0, len(traj_set), step):
            color = 'blue'
            plt.plot(traj_set[i].cartesian.x, traj_set[i].cartesian.y,
                     color=color, zorder=20, linewidth=0.1, alpha=1.0)

    # visualize predictions
    if predictions is not None:
        draw_uncertain_predictions(predictions, rnd.ax)

    # visualize reference path
    if ref_path is not None:
        rnd.ax.plot(ref_path[:, 0], ref_path[:, 1], color='g', marker='.', markersize=1, zorder=19, linewidth=0.8,
                    label='reference path')

    # save as .png file
    if config.debug.save_plots:
        os.makedirs(os.path.join(os.path.dirname(__file__), "../../plots/", str(scenario.scenario_id)),
                    exist_ok=True)
        plot_dir = os.path.join(os.path.dirname(__file__), "../../plots/", str(scenario.scenario_id))
        plt.savefig(f"{plot_dir}/{scenario.scenario_id}_{timestep}.png", format='png', dpi=300,
                    bbox_inches='tight')


    # show plot
    if config.debug.show_plots:
        plt.show(block=True)


def plot_final_trajectory(scenario: Scenario, planning_problem: PlanningProblem, state_list: List[State],
                          config: Configuration, ref_path: np.ndarray = None):
    """
    Function plots occupancies for a given CommonRoad trajectory (of the ego vehicle)
    :param scenario: CommonRoad scenario object
    :param planning_problem CommonRoad Planning problem object
    :param state_list: List of trajectory States
    :param config: Configuration object for plot/save settings
    :param ref_path: Reference path as [(nx2) np.ndarray] (optional)
    :param save_path: Path to save plot as .png (optional)
    """
    # create renderer object (if no existing renderer is passed)
    rnd = MPRenderer(figsize=(20, 10))

    # visualize scenario
    scenario.draw(rnd, draw_params={'time_begin': 0})
    # visualize planning problem
    planning_problem.draw(rnd, draw_params={'planning_problem': {'initial_state': {'state': {
                'draw_arrow': False, "radius": 0.5}}}})
    # visualize occupancies of trajectory
    for state in state_list:
        occ_pos = Rectangle(length=config.vehicle.length, width=config.vehicle.width, center=state.position,
                            orientation=state.orientation)
        occ_pos.draw(rnd, draw_params={'shape': {'rectangle': {'facecolor': '#E37222', 'opacity': 0.6}}})
    # render scenario and occupancies
    rnd.render()

    # visualize reference path
    if ref_path is not None:
        rnd.ax.plot(ref_path[:, 0], ref_path[:, 1], color='g', marker='.', markersize=1, zorder=19, linewidth=0.8,
                    label='reference path')

    # save as .png file
    if config.debug.save_plots:
        os.makedirs(os.path.join(os.path.dirname(__file__), "../../plots/", str(scenario.scenario_id)),
                    exist_ok=True)
        plot_dir = os.path.join(os.path.dirname(__file__), "../../plots/", str(scenario.scenario_id))
        plt.savefig(f"{plot_dir}/{scenario.scenario_id}_final_trajectory.png", format='png', dpi=300,
                    bbox_inches='tight')

    # show plot
    if config.debug.show_plots:
        plt.show(block=True)
