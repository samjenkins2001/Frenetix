__author__ = "Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

from typing import List
from pathlib import Path
import os
import sys
import logging
from cr_scenario_handler.utils.configuration import Configuration
import sqlite3

# KEYS: großbuchstaben
# Logging:
# SCENARIO, AGENT_ID, TIMESTEP
#
# Tabellen:
# Meta: Config
#
# TIMEING:
# Timestep,
#

class SimulationLogger:
    def __init__(self, log_path):

        os.mkdir(log_path, exist_ok=True)
        self.log_path = log_path

        self.con = sqlite3.connect(os.path.join(self.log_path, "simulation.sql"), isolation_level="EXCLUSIVE")
        self.con.executescript("""
            PRAGMA journal_mode = OFF;
            PRAGMA locking_mode = EXCLUSIVE;
            PRAGMA temp_store = MEMORY;
        """)


        self.create_tables()



        os.makedirs(log_path, exist_ok=True)
        self.time_header= None
        self.time_file = "performance_measures.csv"

        # self.set_performance_header()
    # def set_performance_header(self):

    def create_tables(self):
        # Table for main simulation time measurement
        self.con.execute("""
                CREATE TABLE global_performance_measure(
                    scenario TEXT NOT NULL,
                    time_step INT NOT NULL,
                    total_sim_time REAL NOT NULL,
                    global_sim_preprocessing REAL,
                    global_batch_syn REAL,
                    global_visualization REAL,
                    PRIMARY KEY(scenario, time_step)
                   ) STRICT
               """)


        # Table for batch simulation time measurement
        self.con.execute("""
                CREATE TABLE batch_performance_measure(
                    scenario TEXT NOT NULL,
                    batch TEXT NOT NULL,
                    time_step INT NOT NULL,
                    total_sim_time REAL NOT NULL,
                    planning_step_time REAL NOT NULL,
                    syn_time_in REAL,
                    sync_time_out REAL,
                    PRIMARY KEY(scenario, batch, time_step)
                   ) STRICT
               """)

        # TODO von meherern PRozessen beschreibbar? wäre aktueller da kein queing am ende notwendig
        # Table for general information (Scenarios
        self.con.execute("""
            CREATE TABLE meta(
                Scenario TEXT NOT NULL ,
                key TEXT NOT NULL,
                value ANY,
                PRIMARY KEY(Scenario, key)
            ) STRICT
        """)

        # self.con.execute("""
        #         CREATE TABLE agent_evaluation(
        #             scenario TEXT NOT NULL,
        #             time_step INT NOT NULL,
        #             agent_id INT NOT NULL,
        #             original_planning_problem INTEGER,
        #             PRIMARY KEY(scenario, time_step)
        #             )STRICT
        #             """)
        #

    def log_meta(self, scenario, config):
        self.con.execute("INSERT INTO meta Values(?, ?)"(scenario.scenario_id, ))
        self.con.commit()

    def log_global_time(self, time_dict):
        pass

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
