import os
from datetime import datetime
import subprocess
import zipfile
import ruamel.yaml as yaml
import yaml as yml
import numpy as np
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import (
    create_collision_object,
)
from commonroad_dc.pycrcc import ShapeGroup
import commonroad_dc.pycrcc as pycrcc


def calculate_desired_velocity(scenario, planning_problem, state, DT, desired_velocity) -> float:
    try:
        # if the goal is not reached yet, try to reach it
        # get the center points of the possible goal positions
        goal_centers = []
        # get the goal lanelet ids if they are given directly in the planning problem
        if (
                hasattr(planning_problem.goal, "lanelets_of_goal_position")
                and planning_problem.goal.lanelets_of_goal_position is not None
        ):
            goal_lanelet_ids = planning_problem.goal.lanelets_of_goal_position[0]
            for lanelet_id in goal_lanelet_ids:
                lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
                n_center_vertices = len(lanelet.center_vertices)
                goal_centers.append(lanelet.center_vertices[int(n_center_vertices / 2.0)])
        elif hasattr(planning_problem.goal.state_list[0], "position"):
            # get lanelet id of the ending lanelet (of goal state), this depends on type of goal state
            if hasattr(planning_problem.goal.state_list[0].position, "center"):
                goal_centers.append(planning_problem.goal.state_list[0].position.center)
        # if it is a survival scenario with no goal areas, no velocity can be proposed
        elif hasattr(planning_problem.goal.state_list[0], "time_step"):
            if state.time_step > planning_problem.goal.state_list[0].time_step.end:
                return 0.0
            else:
                return planning_problem.initial_state.velocity
        else:
            return 0.0

        distances = []
        for goal_center in goal_centers:
            distances.append(distance(goal_center, state.position))

        # calculate the average distance to the goal positions
        avg_dist = np.mean(distances)

        _, max_remaining_time_steps = calc_remaining_time_steps(
            planning_problem=planning_problem,
            ego_state_time=state.time_step,
            t=0.0,
            dt=DT,
        )
        remaining_time = max_remaining_time_steps * DT

        # if there is time remaining, calculate the difference between the average desired velocity and the velocity of the trajectory
        if remaining_time > 0.0:
            desired_velocity_new = avg_dist / remaining_time
        else:
            desired_velocity_new = 1

    except:
        print("Could not calculate desired velocity")
        desired_velocity_new = desired_velocity

    if np.abs(desired_velocity - desired_velocity_new) > 5:
        if desired_velocity_new > desired_velocity:
            desired_velocity_new = desired_velocity + 2
        else:
            desired_velocity_new = desired_velocity - 2

    return desired_velocity_new


def calc_remaining_time_steps(
    ego_state_time: float, t: float, planning_problem, dt: float
):
    """
    Get the minimum and maximum amount of remaining time steps.

    Args:
        ego_state_time (float): Current time of the state of the ego vehicle.
        t (float): Checked time.
        planning_problem (PlanningProblem): Considered planning problem.
        dt (float): Time step size of the scenario.

    Returns:
        int: Minimum remaining time steps.
        int: Maximum remaining time steps.
    """
    considered_time_step = int(ego_state_time + t / dt)
    if hasattr(planning_problem.goal.state_list[0], "time_step"):
        min_remaining_time = (
            planning_problem.goal.state_list[0].time_step.start - considered_time_step
        )
        max_remaining_time = (
            planning_problem.goal.state_list[0].time_step.end - considered_time_step
        )
        return min_remaining_time, max_remaining_time
    else:
        return False


def create_tvobstacle(
    traj_list: [[float]], box_length: float, box_width: float, start_time_step: int
):
    """
    Return a time variant collision object.

    Args:
        traj_list ([[float]]): List with the trajectory ([x-position, y-position, orientation]).
        box_length (float): Length of the obstacle.
        box_width (float): Width of the obstacle.
        start_time_step (int): Time step of the initial state.

    Returns:
        pyrcc.TimeVariantCollisionObject: Collision object.
    """
    # time variant object starts at the given time step
    tv_obstacle = pycrcc.TimeVariantCollisionObject(time_start_idx=start_time_step)
    for state in traj_list:
        # append each state to the time variant collision object
        tv_obstacle.append_obstacle(
            pycrcc.RectOBB(box_length, box_width, state[2], state[0], state[1])
        )
    return tv_obstacle


def delete_folder(path):
    if os.path.exists(path):
        # shutil.rmtree(path)
        subpr_handle = subprocess.Popen("sudo rm -rf " + path, shell=True)
        wait = subpr_handle.wait()


def delete_file(path):
    if os.path.isfile(path):
        os.remove(path)


def createfolder_if_not_existent(inputpath):
    if not os.path.exists(inputpath):
        os.makedirs(inputpath, mode=0o777)
        name_folder = inputpath.rsplit('/')[-1]
        print("Create " + name_folder + " folder")


def create_time_in_date_folder(inputpath):
    # directory with time stamp to save csv
    date = datetime.now().strftime("%Y_%m_%d")
    time = datetime.now().strftime("%H_%M_%S")
    if not os.path.exists(inputpath):
        os.makedirs(inputpath, mode=0o777)
    if not os.path.exists(os.path.join(inputpath, date)):
        os.makedirs(os.path.join(inputpath, date), mode=0o777)
    os.makedirs(os.path.join(inputpath, date, time), mode=0o777)

    return os.path.join(inputpath, date, time)


def zip_log_files(inputpath):
    filePaths = []
    # Read all directory, subdirectories and file lists
    for root, directories, files in os.walk(inputpath):
        for filename in files:
            # Create the full filepath by using os module.
            filePath = os.path.join(root, filename)
            filePaths.append(filePath)

    # writing files to a zipfile
    zip_file = zipfile.ZipFile(inputpath + '.zip', 'w')
    with zip_file:
        # writing each file one by one
        for file in filePaths:
            zip_file.write(file)

    print(inputpath + '.zip file is created successfully!')

    # Remove Log files
    # shutil.rmtree(inputpath)


def open_config_file(path: str):
    # Load config with the set of tuning parameters
    with open(path) as f:
        config_parameters_ = yml.load(f, Loader=yaml.RoundTripLoader)
    return config_parameters_


def delete_empty_folders(path: str):

    folders = list(os.walk(path))[1:]

    for folder in folders:
        inner_folders = list(os.walk(folder[0]))[1:]
        for inner_folder in inner_folders:
            # folder example: ('FOLDER/3', [], ['file'])
            if not inner_folder[2]:
                os.rmdir(inner_folder[0])
        if not folder[2]:
            try:
                os.rmdir(folder[0])
            except:
                pass


def get_goal_area_shape_group(planning_problem, scenario):
    """
    Return a shape group that represents the goal area.

    Args:
        planning_problem (PlanningProblem): Considered planning problem.
        scenario (Scenario): Considered scenario.

    Returns:
        ShapeGroup: Shape group representing the goal area.
    """
    # get goal area collision object
    # the goal area is either given as lanelets
    if (
        hasattr(planning_problem.goal, "lanelets_of_goal_position")
        and planning_problem.goal.lanelets_of_goal_position is not None
    ):
        # get the polygons of every lanelet
        lanelets = []
        for lanelet_id in planning_problem.goal.lanelets_of_goal_position[0]:
            lanelets.append(
                scenario.lanelet_network.find_lanelet_by_id(
                    lanelet_id
                ).convert_to_polygon()
            )

        # create a collision object from these polygons
        goal_area_polygons = create_collision_object(lanelets)
        goal_area_co = ShapeGroup()
        for polygon in goal_area_polygons:
            goal_area_co.add_shape(polygon)

    # or the goal area is given as positions
    elif hasattr(planning_problem.goal.state_list[0], "position"):
        # get the polygons of every goal area
        goal_areas = []
        for goal_state in planning_problem.goal.state_list:
            goal_areas.append(goal_state.position)

        # create a collision object for these polygons
        goal_area_polygons = create_collision_object(goal_areas)
        goal_area_co = ShapeGroup()
        for polygon in goal_area_polygons:
            goal_area_co.add_shape(polygon)

    # or it is a survival scenario
    else:
        goal_area_co = None

    return goal_area_co


def distance(pos1: np.array, pos2: np.array):
    """
    Return the euclidean distance between 2 points.

    Args:
        pos1 (np.array): First point.
        pos2 (np.array): Second point.

    Returns:
        float: Distance between point 1 and point 2.
    """
    return np.linalg.norm(pos1 - pos2)

# EOF
