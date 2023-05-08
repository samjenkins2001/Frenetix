import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.text import Annotation
import commonroad_rp.occlusion_planning.utils.occ_helper_functions as hf
import numpy as np
from commonroad.visualization.icons import get_obstacle_icon_patch
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType


class OccPlot:
    def __init__(self, config=None, log_path=None, scenario_id=None, occ_scenario=None):
        self.occ_scenario = occ_scenario
        self.scenario_id = scenario_id
        self.log_path = log_path
        self.fast_plot = config.occlusion.use_fast_plot
        self.save_plot = config.occlusion.save_plot
        self.interactive_plot = config.occlusion.interactive_plot
        self.plot_window = config.occlusion.plot_window_dyn
        self.plot_backend = config.occlusion.plot_backend
        self.fig = None
        self.ax = None
        self.occ_cmap = LinearSegmentedColormap.from_list('gr', ["g", "y", "r"], N=10)
        self.time_step = None

        if self.interactive_plot:
            mpl.use(self.plot_backend)
            plt.ion()

    def step_plot(self, time_step=0, ego_state=None, ref_path=None, lanelet_polygon=None, visible_area_vm=None,
                  obstacles=None, visible_area=None, occluded_area=None, additional_plot=None,
                  sidewalk_polygon=None, lanelet_polygon_along_path=None, obstacle_id=False):

        self.time_step = time_step

        if self.ax is None:
            self._create_occ_figure()
        else:
            self.ax.clear()

        self.ax.set_title('Occlusion Map Plot of Timestep {}' .format(time_step))

        self.ax.set_xlim([ego_state.position[0] - 30, ego_state.position[0] + 30])
        self.ax.set_ylim([ego_state.position[1] - 30, ego_state.position[1] + 30])

        ##################
        # Plot Scenario
        ##################

        if ref_path is not None:
            self.ax.plot(ref_path[:, 0], ref_path[:, 1], c='y')

        if lanelet_polygon is not None:
            hf.fill_polygons(self.ax, lanelet_polygon, opacity=0.5, color='gray')
            hf.plot_polygons(self.ax, lanelet_polygon, opacity=0.5, color='k')

        if sidewalk_polygon is not None:
            hf.plot_polygons(self.ax, sidewalk_polygon, opacity=0.5, color='black')

        if lanelet_polygon_along_path is not None:
            hf.fill_polygons(self.ax, lanelet_polygon_along_path, opacity=0.5, color='dimgrey')

        try:
            hf.plot_polygons(self.ax, additional_plot[0], 'c')
            hf.plot_polygons(self.ax, additional_plot[1], 'r')
        except:
            pass

        ##################
        # Plot Ego Vehicle
        ##################

        # plot ego vehicle
        if ego_state is not None:
            try:
                ego_patch = get_obstacle_icon_patch(obstacle_type=ObstacleType('car'),
                                                    pos_x=ego_state.position[0],
                                                    pos_y=ego_state.position[1],
                                                    orientation=ego_state.orientation,
                                                    vehicle_length=5,
                                                    vehicle_width=2,
                                                    vehicle_color='blue',
                                                    edgecolor='black',
                                                    zorder=10)

                self._add_patch(ego_patch)
            except:
                self.ax.plot(ego_state.position[0], ego_state.position[1], 'o', markersize=10)

        ##################
        # Plot visible area from visibility module
        ##################

        if visible_area_vm is not None:
            hf.plot_polygons(self.ax, visible_area_vm, 'g', zorder=2)

        ##################
        # Plot Obstacles
        ##################

        if obstacles is not None:
            for obst in obstacles:

                if obst.pos is None:
                    continue

                # define color
                if obst.visible_at_timestep:
                    color = 'green'
                    alpha = 1
                else:
                    color = "orange"
                    alpha = 0.5

                # create and plot patch
                try:
                    if obst.obstacle_role == "DYNAMIC":
                        pos_x = obst.pos[time_step][0]
                        pos_y = obst.pos[time_step][1]
                        orientation = obst.orientation[time_step]
                    else:
                        pos_x = obst.pos[0]
                        pos_y = obst.pos[1]
                        orientation = obst.orientation

                    if obst.obstacle_type.name == "UNKNOWN":
                        obst_type = ObstacleType('car')
                    else:
                        obst_type = obst.obstacle_type

                    obst_patch = get_obstacle_icon_patch(obstacle_type=obst_type,
                                                         pos_x=pos_x,
                                                         pos_y=pos_y,
                                                         orientation=orientation,
                                                         vehicle_length=obst.obstacle_shape.length,
                                                         vehicle_width=obst.obstacle_shape.width,
                                                         vehicle_color=color,
                                                         edgecolor='black',
                                                         zorder=10)
                    self._add_patch(obst_patch, alpha)

                # plot polygon
                except:
                    hf.fill_polygons(self.ax, obst.polygon, color, zorder=1, opacity=alpha)

                if obstacle_id:
                    # plot obstacle id
                    x = obst.pos_point.x
                    y = obst.pos_point.y
                    self.ax.annotate(obst.obstacle_id, xy=(x, y), xytext=(x, y), zorder=100, color='white')

        ##################
        # Save Plot
        ##################

        if self.save_plot:
            self._save_plot()

        if self.interactive_plot:
            plt.show(block=False)
            plt.pause(0.1)

    def _create_occ_figure(self):
        self.fig, self.ax = plt.subplots()
        self.ax.axis('equal')

    def _save_plot(self):
        plot_dir = os.path.join(self.log_path, "occlusion_plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(f"{plot_dir}/{self.scenario_id}_{self.time_step}.png", format='png', dpi=300)

    def plot_trajectories(self, trajectories, color='k', label=None):
        if trajectories is not None:
            if type(trajectories) == list:
                for traj in trajectories:
                    self.ax.plot(traj.cartesian.x, traj.cartesian.y, color=color)
            else:
                if label is not None:
                    self.ax.plot(trajectories.cartesian.x, trajectories.cartesian.y, label=label)
                    plt.legend(loc="upper left")
                else:
                    self.ax.plot(trajectories.cartesian.x, trajectories.cartesian.y, color=color)

    def debug_trajectory_point_distances(self, occ_map, trajectory, traj_coords, distance, distance_weights):
        self.plot_trajectories(trajectory, 'k')
        self.ax.plot(traj_coords[:, 0], traj_coords[:, 1], 'ko')
        for i, traj in enumerate(traj_coords):
            for j, map_coord in enumerate(occ_map[:, :2]):

                if distance[i, j] > 0:
                    dist = distance[i, j]
                    dist_weight = distance_weights[i, j]
                    x = [traj[0], map_coord[0]]
                    y = [traj[1], map_coord[1]]
                    self.ax.plot(x, y)
                    x = np.sum(x) / 2
                    y = np.sum(y) / 2
                    self.ax.annotate(str(dist) + "-" + str(dist_weight), xy=(x, y), xytext=(x + 0.2, y + 0.2), zorder=100)

    def plot_trajectories_cost_color(self, trajectories, costs, min_costs=None, max_costs=None):
        if min_costs is None:
            min_costs = min(costs)
        if max_costs is None:
            max_costs = max(costs)

        for i, traj in enumerate(trajectories):
            if costs[i] == min_costs:
                self.plot_trajectories(traj, color='c')
            elif costs[i] <= 0.25 * max_costs:
                self.plot_trajectories(traj, color='g')
            elif 0.25 * max_costs < costs[i] <= 0.5 * max_costs:
                self.plot_trajectories(traj, color='y')
            elif 0.5 * max_costs < costs[i] <= 0.75 * max_costs:
                self.plot_trajectories(traj, color='orange')
            else:
                self.plot_trajectories(traj, color='r')
            # print('costs of {} found in trajectory {}' .format(costs[i], i))
        if self.save_plot:
            self._save_plot()

    def plot_trajectory_harm_color(self, trajectories, harm_list, min_harm=0, max_harm=1):
        for traj, harm in zip(trajectories, harm_list):
            if harm == min_harm:
                self.plot_trajectories(traj, color='c')
            elif harm <= 0.25 * max_harm:
                self.plot_trajectories(traj, color='g')
            elif 0.25 * max_harm < harm <= 0.5 * max_harm:
                self.plot_trajectories(traj, color='y')
            elif 0.5 * max_harm < harm <= 0.75 * max_harm:
                self.plot_trajectories(traj, color='orange')
            else:
                self.plot_trajectories(traj, color='r')

    def plot_phantom_collision(self, collision):

        for key in collision:
            if collision[key]['collision']:
                if collision[key]['ego_traj_polygons'] is None:
                    traj = collision[key]['traj']
                    ego_traj_polygons = hf.compute_vehicle_polygons(traj.cartesian.x, traj.cartesian.y,
                                                                    traj.cartesian.theta,
                                                                    self.occ_scenario.ego_width,
                                                                    self.occ_scenario.ego_length)
                    collision[key]['ego_traj_polygons'] = ego_traj_polygons

                hf.draw_collision_trajectory(self.ax, collision[key])

    def plot_phantom_ped_trajectory(self, peds):
        # plot phantom ped trajectories
        for ped in peds:
            hf.fill_polygons(self.ax, ped.polygon, 'gold', zorder=10)
            self.ax.plot(ped.trajectory[:, 0], ped.trajectory[:, 1], 'b')
            self.ax.plot(ped.goal_position[0], ped.goal_position[1], 'bo')

        if self.save_plot:
            self._save_plot()

    def _add_patch(self, patch, alpha=1):
        for p in patch:
            p.set_alpha(alpha)
            self.ax.add_patch(p)

    def plot_uncertainty_map(self, occlusion_map):
        if occlusion_map is not None:
            hf.plot_occ_map(self.ax, occlusion_map, self.occ_cmap)





