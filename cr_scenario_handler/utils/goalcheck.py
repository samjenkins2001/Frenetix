__author__ = "Maximilian Geisslinger, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

"""Provide a class for easy checking if a goal state is reached."""
import copy
import numpy as np
from commonroad_dc.pycrcc import ShapeGroup
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object
import cr_scenario_handler.utils.helper_functions as hf


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


class GoalReachedChecker:
    """GoalChecker for easy checking if the goal is reached."""

    def __init__(self, planning_problem, reference_path, planner):
        """__init__ function."""
        self.goal = planning_problem.goal
        self.goal_state = planning_problem.goal.state_list[0]
        self.status = []
        self.current_time = None
        self.x_cl = None
        self.last_s_position_of_goal = None

        # Initialize to None
        last_pos_in_goal = None
        last_index_in_goal = None

        if "position" in self.goal_state.attributes:
            for index, point in reversed(list(enumerate(reference_path))):
                if self.goal_state.position.contains_point(point):
                    last_pos_in_goal = point
                    last_index_in_goal = index
                    break

            assert last_pos_in_goal is not None, "No Point of the Reference Path is in the Goal Area"

            end_velocity = self.goal_state.velocity.end if "velocity" in self.goal_state.attributes else 10.0
            end_distance = 4.0 * end_velocity  # usual may planning horizon is 4

            if last_pos_in_goal is not None and hf.distance(last_pos_in_goal, reference_path[-1]) <= end_distance:
                deltas = np.diff(reference_path, axis=0)
                distances = np.sqrt((deltas ** 2).sum(axis=1))

                check_distance = 0.0
                buffer_index = last_index_in_goal
                for i in reversed(range(0, last_index_in_goal)):
                    check_distance += distances[i]
                    buffer_index -= 1
                    if check_distance >= end_distance:
                        break

                x_pos = reference_path[buffer_index][0]
                y_pos = reference_path[buffer_index][1]

                if not self.goal_state.position.contains_point(reference_path[buffer_index]):
                    x_pos, y_pos = next(
                        (point for point in reference_path if self.goal_state.position.contains_point(point)),
                        None)

                self.last_goal_position = planner._co.ccosy.convert_to_curvilinear_coords(x_pos, y_pos)[0]

    def register_current_state(self, current_state, x_cl=None):
        """Register the current state and check if in goal."""
        self.status = []
        self.current_time = current_state.time_step
        self.x_cl = x_cl
        for goal_state in self.goal.state_list:
            state_status = {}
            normalized_state = self._normalize_states(current_state, goal_state)
            self._check_position(normalized_state, goal_state, state_status)
            self._check_orientation(normalized_state, goal_state, state_status)
            self._check_velocity(normalized_state, goal_state, state_status)
            self._check_time_step(normalized_state, goal_state, state_status)
            self.status.append(state_status)

    def goal_reached_status(self):
        """Get the goal status."""
        for state_status in copy.deepcopy(self.status):
            if "time_step" in state_status: # time step info available
                timing_flag = state_status.pop("timing_flag")
            if all(list(state_status.values())): # every condition fulfilled
                return True, "Scenario Successful!", state_status
            elif "time_step" in state_status:
                _ = state_status.pop("time_step")
                if (
                    timing_flag == "exceeded"
                    and all(list(state_status.values()))
                ):
                    return True, "Scenario Completed out of Time!", state_status
                elif "position" in state_status:
                    if state_status["position"]:
                        return True, "Scenario Completed Faster than Target Time!", state_status
            if "position" in state_status:
                if self.x_cl[0][0] > self.last_goal_position:
                    return True, "Vehicle Reached Maximum S-Position!", state_status

            # elif "position" in state_status:
            #     if self.last_position_check and not state_status["position"]:
            #         return True, "Scenario Completed Faster than Target Time!", state_status
            # if "position" in state_status:
            #     self.last_position_check = state_status["position"]

            return False, "Scenario is still running!", state_status

    @staticmethod
    def _check_position(normalized_state, goal_state, state_status):
        if hasattr(goal_state, "position"):
            state_status["position"] = goal_state.position.contains_point(
                normalized_state.position
            )

    def _check_orientation(self, normalized_state, goal_state, state_status):
        if hasattr(goal_state, "orientation"):
            state_status["orientation"] = self.goal._check_value_in_interval(
                normalized_state.orientation, goal_state.orientation
            )

    def _check_velocity(self, normalized_state, goal_state, state_status):
        if hasattr(goal_state, "velocity"):
            state_status["velocity"] = self.goal._check_value_in_interval(
                normalized_state.velocity, goal_state.velocity
            )

    def _check_time_step(self, normalized_state, goal_state, state_status):
        if hasattr(goal_state, "time_step"):
            state_status["time_step"] = self.goal._check_value_in_interval(
                normalized_state.time_step, goal_state.time_step
            )
            if normalized_state.time_step > goal_state.time_step.end:
                state_status["timing_flag"] = "exceeded"
            else:
                state_status["timing_flag"] = "not exceeded"

    def _normalize_states(self, current_state, goal_state):
        goal_state_tmp = copy.deepcopy(goal_state)
        goal_state_fields = {
            slot for slot in goal_state.__slots__ if hasattr(goal_state, slot)
        }
        state_fields = {
            slot for slot in goal_state.__slots__ if hasattr(current_state, slot)
        }
        (
            state_new,
            state_fields,
            goal_state_tmp,
            goal_state_fields,
        ) = self.goal._harmonize_state_types(
            current_state, goal_state_tmp, state_fields, goal_state_fields
        )

        if not goal_state_fields.issubset(state_fields):
            raise ValueError(
                "The goal states {} are not a subset of the provided states {}!".format(
                    goal_state_fields, state_fields
                )
            )

        return state_new
