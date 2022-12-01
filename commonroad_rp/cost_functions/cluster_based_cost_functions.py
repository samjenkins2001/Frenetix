import os
import numpy as np
import pickle
from typing import List
from commonroad_rp.trajectories import TrajectorySample


class ClusterBasedCostFunction:

    def __init__(self, rp, configuration):
        self.rp = rp
        self.scenario = rp.scenario
        self.lln = self.scenario.lanelet_network
        # self.configuration = configuration

        model_path = os.path.join(os.getcwd(), "configurations", "cluster", "cluster_model")
        self.model = pickle.load(open(model_path, 'rb'))

        # Enrich ref path data with distances between points and cumulative distances
        ref_path = self.rp.reference_path
        dists = []
        for i in range(1, len(ref_path)):
            dists.append(
                np.sqrt((ref_path[i - 1, 0] - ref_path[i, 0]) ** 2 + (ref_path[i - 1, 1] - ref_path[i, 1]) ** 2))

        dists = np.array(dists).reshape((-1, 1))
        ref_path = np.append(ref_path[1:], dists, axis=1)
        cum_dists = np.array(np.cumsum(ref_path[:, 2]), dtype=float).reshape((-1, 1))
        self.ref_path = np.append(ref_path, cum_dists, axis=1)

        # list of cluster assigmnents
        self.informations = []
        self.cluster_assignments = []

        self.means = np.array(configuration.planning.cluster_means)
        self.stds = np.array(configuration.planning.cluster_stds)

    def evaluate(self, trajectories: List[TrajectorySample], prediction_cost):
        cartesian = trajectories[0]._cartesian

        # get data about the surrounding lanelets
        position = [[cartesian.x[0], cartesian.y[0]]]
        ll_index = self.lln.find_lanelet_by_position(position)[0]
        lanelet = self.lln.find_lanelet_by_id(ll_index[0])

        adj_left = lanelet.adj_left
        adj_left_same_direction = lanelet.adj_left_same_direction

        adj_right = lanelet.adj_right
        adj_right_same_direction = lanelet.adj_right_same_direction

        # INFO list: Returns left and right lanelet info for clustering information
        # Left and right lanelets:           list[0](left) + list[2](right) = Lanelet existent? [0=False, 1=True]
        # Left and right lanelet directions: list[1] + list[3] = Lanelet same direction?, [-1=None, 0=False, 1=True]
        info = []
        # Left lanelet
        if adj_left is None:
            info.append(0)
        else:
            info.append(1)
        if adj_left_same_direction is None:
            info.append(-1)
        elif not adj_left_same_direction:
            info.append(0)
        else:
            info.append(1)
        # Right lanelet
        if adj_right is None:
            info.append(0)
        else:
            info.append(1)
        if adj_right_same_direction is None:
            info.append(-1)
        elif not adj_right_same_direction:
            info.append(0)
        else:
            info.append(1)

        # get acceleration and velocity
        info.append(cartesian.a[0])
        info.append(cartesian.v[0])
        # get prediction data
        mean = np.mean(prediction_cost, axis=0)
        std = np.std(prediction_cost, axis=0)
        info.append(mean)
        info.append(std)

        ref_path_length_for_clustering = 10
        # get reference path data
        s_coord = trajectories[0]._curvilinear.s[0]
        curr_pos = np.argmin(np.absolute(self.ref_path[:, 3] - s_coord))
        fut_pos = np.argmin(np.absolute(self.ref_path[:, 3] - (s_coord + ref_path_length_for_clustering)))
        ref_data = np.linalg.norm(np.cross(self.ref_path[curr_pos - 1, 0:2] - self.ref_path[curr_pos - 2, 0:2],
                                           self.ref_path[curr_pos - 2, 0:2] - self.ref_path[fut_pos - 1, 0:2])) / \
                                           np.linalg.norm(self.ref_path[curr_pos - 1, 0:2] - self.ref_path[curr_pos - 2, 0:2])
        info.append(ref_data)

        # normalize data
        info = np.array(info).reshape((1, -1))
        info = np.subtract(info, self.means)
        info = np.divide(info, self.stds)

        # in case this is the first time step, we need to artificially add the steps that happened before
        if len(self.informations) != 6:
            for i in range(6):
                self.informations.append(info)
        # update list
        self.informations.pop(0)
        self.informations.append(info)

        # get data from the previous steps
        combined_infos = np.append(info, self.informations[3], axis=1)
        combined_infos = np.append(combined_infos, self.informations[0], axis=1)

        # predict cluster assignment
        cluster_assignment = self.model.predict(combined_infos)[0]

        self.cluster_assignments.append(cluster_assignment)
        # print(self.cluster_assignments)

        return cluster_assignment
