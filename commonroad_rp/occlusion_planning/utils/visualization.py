import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.text import Annotation
import commonroad_rp.occlusion_planning.utils.occ_helper_functions as hf
import numpy as np


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

    def step_plot(self, time_step=0, ego_pos=None, ref_path=None, lanelet_polygon=None, visible_area_vm=None,
                  obstacles=None, visible_area=None, occluded_area=None, occlusion_map=None, additional_plot=None,
                  error_flag=False, sidewalk_polygon=None, lanelet_polygon_along_path=None):

        self.time_step = time_step

        if self.ax is None:
            self._create_occ_figure()
        else:
            self.ax.clear()

        self.ax.set_title('Occlusion Map Plot of Timestep {}' .format(time_step))

        if error_flag:
            self.ax.text(0, 0, "Could not create plot, because occlusion map could not be initialized!",
                         ha='center', va='center', fontsize=12, color='red')
            self.ax.axis('off')
        else:

            if occlusion_map is not None:
                self.ax.set_xlim([min(occlusion_map[:, 1] - self.plot_window),
                                  max(occlusion_map[:, 1] + self.plot_window)])
                self.ax.set_ylim([min(occlusion_map[:, 2] - self.plot_window),
                                  max(occlusion_map[:, 2] + self.plot_window)])
            else:
                self.ax.set_xlim([ego_pos[0] - 30, ego_pos[0] + 30])
                self.ax.set_ylim([ego_pos[1] - 30, ego_pos[1] + 30])

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

            if ego_pos is not None:
                self.ax.plot(ego_pos[0], ego_pos[1], 'o', markersize=10)

            if visible_area_vm is not None:
                hf.plot_polygons(self.ax, visible_area_vm, 'g', zorder=2)

            if obstacles is not None:
                for obst in obstacles:
                    if obst.polygon is not None:
                        if obst.visible_at_timestep:
                            color = 'c'
                        else:
                            color = 'm'
                        x = obst.pos_point.x
                        y = obst.pos_point.y
                        self.ax.annotate(obst.obstacle_id, xy=(x, y), xytext=(x + 0.2, y + 0.2), zorder=100)
                        hf.fill_polygons(self.ax, obst.polygon, color, zorder=1)

            if self.fast_plot:
                if visible_area is not None:
                    hf.fill_polygons(self.ax, visible_area, 'g')
                if occluded_area is not None:
                    hf.fill_polygons(self.ax, occluded_area, 'r')
            else:
                if occlusion_map is not None:
                    hf.plot_occ_map(self.ax, occlusion_map, self.occ_cmap)

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

    def plot_trajectories_cost_color(self, trajectories, costs, length=None, mean_v_occ=None):
        min_costs = min(costs)
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
            #print('costs of {} found in trajectory {}' .format(costs[i], i))
        if self.save_plot:
            self._save_plot()

    def plot_phantom_collision(self, collision):

        if collision['ego_traj_polygons'] is None:
            traj = collision['traj']
            ego_traj_polygons = hf.compute_vehicle_polygons(traj.cartesian.x, traj.cartesian.y,
                                                            traj.cartesian.theta,
                                                            self.occ_scenario.ego_width,
                                                            self.occ_scenario.ego_length)
            collision['ego_traj_polygons'] = ego_traj_polygons

        hf.draw_collision_trajectory(self.ax, collision)





