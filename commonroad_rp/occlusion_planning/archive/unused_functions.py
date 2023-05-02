"""
This file contains functions that were created in the development process and are no longer needed.
However, they are executable and could be used again in the later course.

Author: Korbinian Moller, TUM
Date: 19.04.2023
"""


# function is used to merge the collision dict from _check_trajectory_collision with the output of the harm model
def merge_harm_collision_dict(collision, ego_harm_traj, obst_harm_traj) -> dict:

    if collision is not None:
        for key in collision:
            if key in ego_harm_traj and key in obst_harm_traj:
                collision[key]['ego_harm_traj'] = ego_harm_traj[key]
                collision[key]['obst_harm_traj'] = obst_harm_traj[key]

        return collision


