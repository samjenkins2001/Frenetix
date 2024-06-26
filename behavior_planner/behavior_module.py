__author__ = "Moritz Ellermann, Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

# general imports
import copy
import os
import time

# commonroad imports

# project imports
import behavior_planner.utils.helper_functions as hf
from behavior_planner.utils.helper_logging import BehaviorLogger
from behavior_planner.utils.velocity_planner import VelocityPlanner
from behavior_planner.utils.path_planner import PathPlanner
from behavior_planner.utils.FSM_model import EgoFSM

# TODO: reorganize the State classes into Dataclasses


class BehaviorModule(object):
    """
    Behavior Module: Coordinates Path Planner, Velocity Planner and Finite State Machine (FSM) to determine the
    reference path and desired velocity for the reactive planner.

    TODO: Include FSM
    """

    def __init__(self, scenario, planning_problem, init_ego_state, config, log_path):
        """ Init Behavior Module.

        Args:
        pro_path (str): project path.
        scenario: scenario.
        init_ego_state : initialized ego state.
        """

        self.BM_state = BehaviorModuleState()  # behavior module information

        # load config
        self.BM_state.config = config
        self.behavior_config = config.behavior
        self.behavior_config.behavior_log_path_scenario = os.path.join(log_path, "behavior_logs")

        os.makedirs(self.behavior_config.behavior_log_path_scenario, exist_ok=True)

        # init behavior planner and load scenario information
        self.VP_state = self.BM_state.VP_state  # velocity planner information
        self.PP_state = self.BM_state.PP_state  # path planner information
        self.FSM_state = self.BM_state.FSM_state  # FSM information
        self.BM_state.vehicle_params = config.vehicle
        self.BM_state.init_velocity = init_ego_state.velocity
        self.BM_state.dt = config.behavior.dt

        self.BM_state.scenario = scenario
        self.BM_state.planning_problem = planning_problem

        self.BM_state.country = hf.find_country_traffic_sign_id(self.BM_state.scenario)
        self.BM_state.current_lanelet_id, self.BM_state.speed_limit, self.BM_state.street_setting = \
            hf.get_lanelet_information(
                scenario=self.BM_state.scenario,
                reference_path_ids=[],
                ego_state=init_ego_state,
                country=self.BM_state.country)

        # init path planner
        self.path_planner = PathPlanner(BM_state=self.BM_state)
        self.path_planner.execute_route_planning()
        self._retrieve_lane_changes_from_navigation()

        # init ego FSM
        self.ego_FSM = EgoFSM(BM_state=self.BM_state)
        self.FSM_state = self.ego_FSM.FSM_state

        # init velocity planner
        self.velocity_planner = VelocityPlanner(BM_state=self.BM_state)

        # initialize loggers
        self.behavior_logger = BehaviorLogger(self.behavior_config)
        self.behavior_message_logger = self.behavior_logger.message_logger

        # outputs
        self.behavior_output = BehaviorOutput(self.BM_state)
        self.reference_path = self.BM_state.PP_state.reference_path
        self.desired_velocity = None
        self.stop_point = {
            "pos_s": self.BM_state.ref_position_s,
            "velocity": self.BM_state.init_velocity
        }
        self.flags = {"stopping_for_traffic_light": None,
                      "waiting_for_green_light": None
                      }

        # logging
        self.behavior_message_logger.critical("Behavior Module initialized")
        self.behavior_message_logger.debug("simulating scenario: " + str(self.BM_state.scenario.scenario_id))

    def execute(self, predictions, ego_state, time_step):
        """ Execute behavior module.

        TODO: Dynamische Entscheidungen in jedem Time step, highlevel (y.B. Spurwechsel) nur alle 300 - 500 ms

        Args:
        predictions (dict): current predictions.
        ego_state (List): current state of ego vehicle.

        return: behavior_output (BehaviorOutput): Class holding all information for Reactive Planner
        """
        # if (ego_state.time_step > 0 and
        #         not (time_step / self.behavior_config.replanning_frequency == 1 or
        #              self.behavior_config.replanning_frequency < 2)):
        #     return copy.deepcopy(self.behavior_output)

        # start timer
        timer = time.time()

        # inputs
        self.BM_state.predictions = predictions
        self.BM_state.ego_state = ego_state
        self.BM_state.time_step = ego_state.time_step  # time_step

        self._get_ego_position(ego_state)

        self.BM_state.future_factor = int(self.BM_state.ego_state.velocity // 4) + 1  # for lane change maneuvers
        self._collect_necessary_information()

        # execute velocity planner
        self.velocity_planner.execute()
        self.desired_velocity = self.VP_state.desired_velocity

        # execute FSM
        self.ego_FSM.execute()

        # execute path planner
        if self.FSM_state.do_lane_change:
            self.path_planner.execute_lane_change()
        if self.FSM_state.undo_lane_change:
            self.path_planner.undo_lane_change()
        self.reference_path = self.PP_state.reference_path

        # calculate stopping points
        self._calculate_stopping_point()

        # update behavior flags
        self.flags["stopping_for_traffic_light"] = self.FSM_state.slowing_car_for_traffic_light
        self.flags["waiting_for_green_light"] = self.FSM_state.waiting_for_green_light

        # update behavior output; input for reactive planner
        self.behavior_output.reference_path = self.reference_path
        self.behavior_output.desired_velocity = self.desired_velocity
        self.behavior_output.stop_point_s = self.stop_point.get("pos_s")
        self.behavior_output.desired_velocity_stop_point = self.stop_point.get("velocity")
        self.behavior_output.behavior_planner_state = self.BM_state.BP_state.set_values(self.BM_state)

        # end timer
        timer = time.time() - timer

        # set current States vor Visualization TODO change to update visuals via passed object
        self.behavior_config.behavior_state_static = self.FSM_state.behavior_state_static
        self.behavior_config.situation_state_static = self.FSM_state.situation_state_static
        self.behavior_config.behavior_state_dynamic = self.FSM_state.behavior_state_dynamic
        self.behavior_config.situation_state_dynamic = self.FSM_state.situation_state_dynamic

        # logging
        self.behavior_message_logger.debug("VP velocity mode: " + str(self.VP_state.velocity_mode))
        self.behavior_message_logger.debug("VP TTC velocity: " + str(self.VP_state.TTC))
        self.behavior_message_logger.debug("VP MAX velocity: " + str(self.VP_state.MAX))
        if self.VP_state.closest_preceding_vehicle is not None:
            self.behavior_message_logger.debug("VP position of preceding vehicle: " + str(self.VP_state.closest_preceding_vehicle.get('pos_list')[0]))
            self.behavior_message_logger.debug("VP velocity of preceding vehicle: " + str(self.VP_state.vel_preceding_veh))
            self.behavior_message_logger.debug("VP distance to preceding vehicle: " + str(self.VP_state.dist_preceding_veh))
            self.behavior_message_logger.debug("VP safety distance to preceding vehicle: " + str(self.VP_state.safety_dist))
        self.behavior_message_logger.debug("VP recommended velocity: " + str(self.VP_state.goal_velocity))
        self.behavior_message_logger.debug("BP recommended desired velocity: " + str(self.desired_velocity))
        self.behavior_message_logger.debug("current ego velocity: " + str(self.BM_state.ego_state.velocity))
        self.behavior_message_logger.info(f"Behavior Planning Time: \t\t{timer:.5f} s")

        self.behavior_logger.log_data(self.BM_state.__dict__)

        return copy.deepcopy(self.behavior_output)

    def _retrieve_lane_changes_from_navigation(self):
        self.BM_state.nav_lane_changes_left = 0
        self.BM_state.nav_lane_changes_right = 0
        lane_change_instructions = hf.retrieve_glb_nav_path_lane_changes(self.BM_state.scenario, self.BM_state.global_nav_route)
        for idx, instruction in enumerate(lane_change_instructions):
            if lane_change_instructions[idx] == 1:
                lanelet = self.BM_state.scenario.lanelet_network.find_lanelet_by_id(
                    self.BM_state.global_nav_route.list_ids_lanelets[idx])
                if lanelet.adj_left == self.BM_state.global_nav_route.list_ids_lanelets[idx + 1]:
                    self.BM_state.nav_lane_changes_left += 1
                if lanelet.adj_right == self.BM_state.global_nav_route.list_ids_lanelets[idx + 1]:
                    self.BM_state.nav_lane_changes_right += 1

    def _get_ego_position(self, ego_state):
        try:
            self.BM_state.ref_position_s = self.PP_state.cl_ref_coordinate_system.convert_to_curvilinear_coords(
                ego_state.position[0], ego_state.position[1])[0]
        except:
            self.behavior_message_logger.error("Ego position out of reference path coordinate system projection domain")
        try:
            self.BM_state.ref_position_s = self.PP_state.cl_nav_coordinate_system.convert_to_curvilinear_coords(
                ego_state.position[0], ego_state.position[1])[0]
        except:
            self.behavior_message_logger.error("Ego position out of navigation route coordinate system projection domain")

    def _collect_necessary_information(self):
        self.BM_state.current_lanelet_id, self.BM_state.speed_limit, self.BM_state.street_setting_scenario = \
            hf.get_lanelet_information(
                scenario=self.BM_state.scenario,
                reference_path_ids=self.PP_state.reference_path_ids,
                ego_state=self.BM_state.ego_state,
                country=self.BM_state.country)

        self.BM_state.current_lanelet = \
            self.BM_state.scenario.lanelet_network.find_lanelet_by_id(self.BM_state.current_lanelet_id)

    def _calculate_stopping_point(self):
        """
        Calculates the point up to which the reactive planner should plan
        The point consists of two parts:
        1. The position relative to the reference path (S-position)
        2. The desired velocity at that point
        \n
        If a preceding vehicle exists plan to the point, where the preceding vehicle would come to a standstill in case
        of a breaking scenario and aim to have the same velocity as the preceding vehicle or when stopping for a static
        goal, aim to be at a standstill at the static goal s_stopping_position.
        If there is no preceding vehicle and a static goal with a stopping point is scheduled, plan to the static goal
        s_stopping_position and aim to have no velocity there. If there is no stop planned choose the maximal distance
        the current_velocity * default_time_horizon, the comfortable_stopping_distance and eventually the stop_point_s
        of the current static goal
        """
        # self.stop_point = {
        #     "position_s": None,
        #     "velocity": None
        # }
        if self.VP_state.TTC is not None:
            # car in front of ego vehicle
            ttc_stopping_stop_point_s = (self.BM_state.ref_position_s
                                         + self.VP_state.dist_preceding_veh
                                         + self.VP_state.stop_dist_preceding_veh
                                         - self.VP_state.min_safety_dist)
            if self.FSM_state.behavior_state_static in ["TrafficLight", "Crosswalk", "StopSign", "YieldSign"]:
                if self.ego_FSM.FSM_street_setting.cur_state.FSM_static.cur_state.cur_state.startswith("Waiting"):
                    # Don't move: WaitingForGreenLight, WaitingForCrosswalkClearance,
                    # WaitingForStopYieldSignClearance, WaitingForTurnClearance
                    self.stop_point["pos_s"] = self.BM_state.ref_position_s
                    self.stop_point["velocity"] = 0.0
                elif self.ego_FSM.FSM_street_setting.cur_state.FSM_static.cur_state.cur_state == "Stopping":
                    # set stop point to nearest from static goal or detected vehicle
                    if ttc_stopping_stop_point_s >= self.BM_state.current_static_goal.stop_point_s:
                        # aim to stop at the stop point of the traffic light, crosswalk or traffic sign
                        self.stop_point["pos_s"] = self.BM_state.current_static_goal.stop_point_s
                        self.stop_point["velocity"] = 0.0
                    else:
                        # car in front of ego vehicle, stop behind the car in front
                        self.stop_point["pos_s"] = ttc_stopping_stop_point_s
                        # don't accelerate during stopping
                        self.stop_point["velocity"] = min(self.VP_state.vel_preceding_veh, self.stop_point["velocity"])
                else:
                    # use TTC as measure for stopping point calculation
                    self.stop_point["pos_s"] = ttc_stopping_stop_point_s
                    self.stop_point["velocity"] = self.VP_state.vel_preceding_veh
            else:
                # use TTC as measure for stopping point calculation
                self.stop_point["pos_s"] = ttc_stopping_stop_point_s
                self.stop_point["velocity"] = self.VP_state.vel_preceding_veh
        else:
            # no car in front of ego vehicle
            if self.FSM_state.behavior_state_static in ["PrepareTrafficLight", "TrafficLight",
                                                        "PrepareCrosswalk", "Crosswalk",
                                                        "PrepareYieldSign", "YieldSign",
                                                        "PrepareStopSign", "StopSign"]:
                # Situation States of 'Prepare' behavior states
                if self.ego_FSM.FSM_street_setting.cur_state.FSM_static.cur_state.cur_state.startswith("Observing"):
                    # ObservingTrafficLight, ObservingCrosswalk, ObservingStopYieldSign
                    self.stop_point["pos_s"] = self.BM_state.current_static_goal.stop_point_s
                    self.stop_point["velocity"] = self.VP_state.goal_velocity
                elif self.ego_FSM.FSM_street_setting.cur_state.FSM_static.cur_state.cur_state == "SlowingDown":
                    self.stop_point["pos_s"] = self.BM_state.current_static_goal.stop_point_s
                    self.stop_point["velocity"] = 0.0
                # Situation States of behavior states
                elif self.ego_FSM.FSM_street_setting.cur_state.FSM_static.cur_state.cur_state == "GreenLight":
                    self.stop_point["pos_s"] = max(
                        self.BM_state.current_static_goal.stop_point_s,
                        self.VP_state.comfortable_stopping_distance,
                        self.BM_state.ego_state.velocity * self.behavior_config.default_time_horizon
                    )
                    self.stop_point["velocity"] = self.VP_state.goal_velocity
                elif self.ego_FSM.FSM_street_setting.cur_state.FSM_static.cur_state.cur_state.endswith("Clear"):
                    # CrosswalkClear, StopYieldSignClear, TurnClear
                    self.stop_point["pos_s"] = max(
                        self.BM_state.current_static_goal.stop_point_s,
                        self.VP_state.comfortable_stopping_distance,
                        self.BM_state.ego_state.velocity * self.behavior_config.default_time_horizon
                    )
                    self.stop_point["velocity"] = self.VP_state.goal_velocity
                elif self.ego_FSM.FSM_street_setting.cur_state.FSM_static.cur_state.cur_state == "Stopping":
                    self.stop_point["pos_s"] = self.BM_state.current_static_goal.stop_point_s
                    self.stop_point["velocity"] = 0.0
                elif self.ego_FSM.FSM_street_setting.cur_state.FSM_static.cur_state.cur_state.startswith("Waiting"):
                    # Don't move: WaitingForGreenLight, WaitingForCrosswalkClearance,
                    # WaitingForStopYieldSignClearance, WaitingForTurnClearance
                    self.stop_point["pos_s"] = self.BM_state.ref_position_s
                    self.stop_point["velocity"] = 0.0
                elif self.ego_FSM.FSM_street_setting.cur_state.FSM_static.cur_state.cur_state == "ContinueDriving":
                    self.stop_point["pos_s"] = max(
                        self.VP_state.comfortable_stopping_distance,
                        self.BM_state.ego_state.velocity * self.behavior_config.default_time_horizon
                    )
                    self.stop_point["velocity"] = self.VP_state.goal_velocity
                else:
                    self.behavior_message_logger.warning(
                        f"'{self.ego_FSM.FSM_street_setting.cur_state.FSM_static.cur_state.cur_state}'"
                        f" is not a valid Situation State for {self.FSM_state.behavior_state_static}")
            else:
                self.stop_point["pos_s"] = max(
                    self.VP_state.comfortable_stopping_distance,
                    self.BM_state.ego_state.velocity * self.behavior_config.default_time_horizon
                )
                self.stop_point["velocity"] = self.VP_state.goal_velocity
        self.stop_point["pos_s"] -= self.BM_state.vehicle_params.length / 2


class BehaviorModuleState(object):
    """Behavior Module State class containing all information the Behavior Module is working with."""

    def __init__(self):
        # general
        self.vehicle_params = None
        self.country = None
        self.scenario = None
        self.planning_problem = None
        self.priority_right = None

        # Behavior Module inputs
        self.ego_state = None
        self.predictions = None
        self.time_step = None
        self.dt = None

        # FSM and Velocity Planner information
        self.FSM_state = FSMState()
        self.VP_state = VelocityPlannerState()
        self.PP_state = PathPlannerState()
        self.BP_state = BehaviorPlannerState()

        # Behavior Module information
        self.street_setting = None
        self.ref_position_s = None
        self.current_lanelet_id = None
        self.current_lanelet = None
        self.current_static_goal = None

        # velocity
        self.init_velocity = None
        self.speed_limit = None

        # navigation
        self.global_nav_route = None
        self.nav_lane_changes_left = 0
        self.nav_lane_changes_right = 0
        self.overtaking = None


class FSMState(object):
    """FSM state class containing all information the Finite State Machine is working with."""

    def __init__(self):
        # street setting state
        self.street_setting = None

        # static behavior states
        self.behavior_state_static = None
        self.situation_state_static = None

        # dynamic behavior states
        self.behavior_state_dynamic = None
        self.situation_state_dynamic = None

        # time_step_counter
        self.situation_time_step_counter = None

        # vehicle status
        self.detected_lanelets = None

        # information
        self.lane_change_target_lanelet_id = None
        self.lane_change_target_lanelet = None
        self.obstacles_on_target_lanelet = None

        # information flags
        self.overtake_lange_changes_offset = None  # number of lane changes that need to be undone after overtaking; could also be impelmented differently by checking the the lane changes that the new reference path is giving

        # free space offset
        self.free_space_offset = 0
        self.change_velocity_for_lane_change = None

        # permission flags
        self.free_space_on_target_lanelet = None

        self.lane_change_left_ok = None
        self.lane_change_right_ok = None
        self.lane_change_left_done = None
        self.lane_change_right_done = None

        self.lane_change_prep_right_abort = None
        self.lane_change_prep_left_abort = None
        self.lane_change_right_abort = None
        self.lane_change_left_abort = None

        self.no_auto_lane_change = None

        self.turn_clear = None
        self.crosswalk_clear = None
        self.stop_yield_sign_clear = None

        # action flags
        self.do_lane_change = None
        self.undo_lane_change = None

        # reaction flags
        self.initiated_lane_change = None
        self.undid_lane_change = None

        # traffic light
        self.traffic_light_state = None
        self.slowing_car_for_traffic_light = None
        self.waiting_for_green_light = None


class VelocityPlannerState(object):
    """Velocity Planner State class containing all information the Velocity Planner is working with."""

    def __init__(self):
        # outputs
        self.desired_velocity = None
        self.goal_velocity = None
        self.velocity_mode = None

        # general
        self.ttc_norm = 8
        self.speed_limit_default = None
        self.TTC = None
        self.MAX = None
        self.comfortable_stopping_distance = None

        # TTC velocity
        self.closest_preceding_vehicle = None
        self.dist_preceding_veh = None
        self.vel_preceding_veh = None
        self.ttc_conditioned = None
        self.ttc_relative = None  # optimal relative velocity to the preceding vehicle
        self.stop_dist_preceding_veh = None
        self.min_safety_dist = None
        self.safety_dist = None

        # conditions
        self.condition_factor = None  # factor to express driving conditions of the vehicle; ∈ [0,1]
        self.lon_dyn_cond_factor = None  # factor to express longitudinal driving conditions; ∈ [0,1]
        self.lat_dyn_cond_factor = None  # factor to express lateral driving; ∈ [0,1]
        self.visual_cond_factor = None  # factor to express visual driving conditions; ∈ [0,1]

        # traffic light
        self.stop_distance = None
        self.dist_to_tl = None


class PathPlannerState(object):
    """Velocity Planner State class containing information about the Path Planner State"""

    def __init__(self):
        self.static_route_plan = None
        self.route_plan_ids = None
        self.reference_path = None
        self.reference_path_ids = None
        self.cl_ref_coordinate_system = None
        self.cl_nav_coordinate_system = None


class BehaviorPlannerState(object):
    """ Behavior Planner State class

    This class is holding all externally relevant information of the Behavior Planner
    """

    def __init__(self):
        # FSM States
        self.street_setting = None  # string

        self.behavior_state_static = None  # string
        self.situation_state_static = None  # string

        self.behavior_state_dynamic = None  # string
        self.situation_state_dynamic = None  # string

        self.lane_change_target_lanelet_id = None  # string

        # Velocity Planner
        self.velocity = None  # float
        self.goal_velocity = None  # float
        self.desired_velocity = None  # float  # also passed separately
        self.TTC = None  # float
        self.MAX = None  # float

        self.slowing_car_for_traffic_light = None  # boolean
        self.waiting_for_green_light = None  # boolean

        self.condition_factor = None  # factor to express driving conditions of the vehicle; ∈ [0,1]
        self.lon_dyn_cond_factor = None  # factor to express longitudinal driving conditions; ∈ [0,1]
        self.lat_dyn_cond_factor = None  # factor to express lateral driving; ∈ [0,1]
        self.visual_cond_factor = None  # factor to express visual driving conditions; ∈ [0,1]

        # Path Planner
        # self.reference_path = None  # list of tuples of floats  # just passed separately
        self.reference_path_ids = None  # list of strings

    def set_values(self, BM_state):
        """sets all values of this class, so that BM_state will not be part of the dict of this class, and returns a deepcopy"""

        # FSM States
        self.street_setting = BM_state.FSM_state.street_setting  # string

        self.behavior_state_static = BM_state.FSM_state.behavior_state_static  # string
        self.situation_state_static = BM_state.FSM_state.situation_state_static  # string

        self.behavior_state_dynamic = BM_state.FSM_state.behavior_state_dynamic  # string
        self.situation_state_dynamic = BM_state.FSM_state.situation_state_dynamic  # string

        self.lane_change_target_lanelet_id = BM_state.FSM_state.lane_change_target_lanelet_id  # string

        self.slowing_car_for_traffic_light = BM_state.FSM_state.slowing_car_for_traffic_light  # boolean
        self.waiting_for_green_light = BM_state.FSM_state.waiting_for_green_light  # boolean

        # Velocity Planner
        self.velocity = BM_state.ego_state.velocity if BM_state.ego_state is not None else BM_state.init_velocity  # float
        self.goal_velocity = BM_state.VP_state.goal_velocity  # float
        self.desired_velocity = BM_state.VP_state.desired_velocity  # float
        self.TTC = BM_state.VP_state.TTC  # float
        self.MAX = BM_state.VP_state.MAX  # float

        self.condition_factor = BM_state.VP_state.condition_factor  # ∈ [0,1]
        self.lon_dyn_cond_factor = BM_state.VP_state.lon_dyn_cond_factor  # ∈ [0,1]
        self.lat_dyn_cond_factor = BM_state.VP_state.lat_dyn_cond_factor  # ∈ [0,1]
        self.visual_cond_factor = BM_state.VP_state.visual_cond_factor  # ∈ [0,1]

        # Path Planner
        # self.reference_path = BM_state.PP_state.reference_path  # list of tuples of floats  # passed separately
        self.reference_path_ids = BM_state.PP_state.reference_path_ids  # list of strings

        return copy.deepcopy(self.__dict__)


class BehaviorOutput(object):
    """Class for collected Behavior Input for Reactive Planner"""

    def __init__(self, BM_state):
        self.desired_velocity = None
        self.reference_path = None
        self.stop_point_s = None
        self.desired_velocity_stop_point = None
        self.behavior_planner_state = BM_state.BP_state.set_values(BM_state)  # deepcopies values
