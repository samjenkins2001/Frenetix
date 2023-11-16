__author__ = "Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

from typing import List
import os
import sys
import logging
from cr_scenario_handler.utils.configuration import Configuration


def init_log(log_path: str):
    """Create log file for simulation-level logging and write the header.

    The log file will contain the following fields:
    time_step: The time step of this log entry.
    domain_time: The time step expressed in simulated time.
    total_planning_time: The wall clock time required for executing the planner step of all agents.
    total_synchronization_time: The wall clock time required for synchronizing the agents.
    agent_ids: A list of the IDs of all agents in the simulation.
    agent_states: A list of the return values of the agents' step function,
        in the same order as the agent_ids

    :param log_path: Base path the log file is written to
    """

    os.makedirs(log_path, exist_ok=True)

    with open(os.path.join(log_path, "execution_logs.csv"), "w+") as log_file:
        log_file.write("time_step;domain_time;total_planning_time;total_synchronization_time;agent_ids;agent_states;")


def append_log(log_path: str, time_step: int, domain_time: float, total_planning_time: float,
               total_synchronization_time: float, agent_ids: List[int], agent_states: List[int]):
    """Write the log entry for one simulation step.

    :param log_path: Path to the directory containing the log file
    :param time_step: Number of the current time step
    :param domain_time: Current time inside the simulation
    :param total_planning_time: Wall clock time for stepping all agents
    :param total_synchronization_time: Wall clock time for exchanging dummy obstacles
    :param agent_ids: List of all agent ids in the scenario
    :param agent_states: Return codes from all agents
    """

    entry = "\n"
    entry += str(time_step) + ";"
    entry += str(domain_time) + ";"
    entry += str(total_planning_time) + ";"
    entry += str(total_synchronization_time) + ";"
    entry += str(agent_ids) + ";"
    entry += str(agent_states) + ";"

    with open(os.path.join(log_path, "execution_logs.csv"), "a") as log_file:
        log_file.write(entry)


def logger_initialization(config: Configuration, log_path, logger = "Simulation_logger") -> logging.Logger:
    """
    Message Logger Initialization
    """

    # msg logger
    msg_logger = logging.getLogger(logger) # logging.getLogger("Simulation_logger")

    if msg_logger.handlers:
        return msg_logger

    # Create directories
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # create file handler (outputs to file)
    path_log = os.path.join(log_path, "messages.log")
    file_handler = logging.FileHandler(path_log)

    # set logging levels
    loglevel = config.debug.msg_log_mode if hasattr(config, "debug") else config.simulation.msg_log_mode

    msg_logger.setLevel(loglevel)
    file_handler.setLevel(loglevel)

    # create log formatter
    # formatter = logging.Formatter('%(asctime)s\t%(filename)s\t\t%(funcName)s@%(lineno)d\t%(levelname)s\t%(message)s')
    log_formatter = logging.Formatter("%(levelname)-8s [%(asctime)s] --- %(message)s (%(filename)s:%(lineno)s)",
                                  "%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(log_formatter)

    # create stream handler (prints to stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(loglevel)

    # create stream formatter
    stream_formatter = logging.Formatter("%(levelname)-8s [%(filename)s]: %(message)s")
    stream_handler.setFormatter(stream_formatter)

    # add handlers
    msg_logger.addHandler(file_handler)
    msg_logger.addHandler(stream_handler)
    msg_logger.propagate = False

    return msg_logger
