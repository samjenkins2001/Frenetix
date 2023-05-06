from typing import List

from commonroad.geometry.shape import Rectangle
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.trajectory import Trajectory

from commonroad_prediction.prediction_module import PredictionModule
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.state import CustomState, State

from commonroad.scenario.scenario import Scenario

from commonroad_rp.configuration import Configuration, VehicleConfiguration
from commonroad_rp import prediction_helpers as ph


TIMEOUT = 20
"""Timeout value used when waiting for messages during parallel execution"""


def get_all_obstacle_ids(scenario: Scenario) -> list:
    """Return all obstacle IDs that exist in a Commonroad Scenario

    :param scenario: The scenario of Commonroad we use
    """
    return [obs.obstacle_id for obs in scenario.obstacles]


def get_predictions(config: Configuration, predictor: PredictionModule,
                    scenario: Scenario, current_timestep: int):
    """ Calculate the predictions for all obstacles in the scenario.

    :param config: The configuration.
    :param predictor: The prediction module object to use.
    :param scenario: The scenario for which to calculate the predictions.
    :param current_timestep: The timestep after which to start predicting.
    """

    if config.prediction.mode:
        if config.prediction.mode == "walenet":
            # Calculate predictions for all obstacles using WaleNet.
            # The state is only used for accessing the current timestep.
            predictions = ph.main_prediction(predictor, scenario, [obs.obstacle_id for obs in scenario.obstacles],
                                             CustomState(time_step=current_timestep), scenario.dt,
                                             [float(config.planning.planning_horizon)])
        elif config.prediction.mode == "lanebased":
            print("Lane-based predictions are not supported for multiagent simulations.")
            predictions = None
        else:
            predictions = None
    else:
        predictions = None

    return predictions


def trajectory_to_obstacle(state_list: List[State],
                           vehicle_params: VehicleConfiguration, obstacle_id: int):
    """ Convert a state to a dummy DynamicObstacle object.

    :param state_list: List of all states in the trajectory.
    :param vehicle_params: VehicleConfiguration containing the shape of the obstacle.
    :param obstacle_id: ID to be assigned, usually equal to the agent ID.

    :return: A DynamicObstacle with the given state list as trajectory.
    """

    trajectory = Trajectory(initial_time_step=state_list[0].time_step, state_list=state_list)
    shape = Rectangle(vehicle_params.length, vehicle_params.width)
    prediction = TrajectoryPrediction(trajectory, shape)

    return DynamicObstacle(obstacle_id, ObstacleType.CAR, shape, trajectory.state_list[0], prediction)
