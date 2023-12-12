__author__ = "Rainer Trauth"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

import json
from typing import List

import os
import sys
import logging
from omegaconf import DictConfig, ListConfig

from cr_scenario_handler.utils.configuration import Configuration
import sqlite3

from cr_scenario_handler.utils.agent_status import AgentStatus, TIMEOUT

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

    @staticmethod
    def _convert_config(config: Configuration) -> dict:
        data = dict()
        for item in config.__dict__:
            # print(item)
            ii = getattr(config, item)
            data[item] = dict()
            for it in ii.__dict__:
                val = ii.__dict__[it]
                if isinstance(val, DictConfig):
                    val = dict(val)
                if isinstance(val, ListConfig):
                    val = list(val)

                # print(f"name={it} type={type(val)}")

                data[item][it] = val

        return data

    def __init__(self, config):


        self.config = config
        self.eval_conf = config.evaluation
        self.scenario = self.config.simulation.name_scenario
        self.original_planning_problem_id = None
        log_path = self.config.simulation.log_path
        self.log_path = log_path if self.scenario not in log_path else log_path.replace(self.scenario, "")
        self.log_time = self.eval_conf.evaluate_runtime
        self.scenario = self.config.simulation.name_scenario

        os.makedirs(self.log_path, exist_ok=True)

        self.con = sqlite3.connect(os.path.join(self.log_path, "simulation.db"), timeout=TIMEOUT,
                                                isolation_level="EXCLUSIVE"
                                   )

        self.con.executescript("""
            PRAGMA journal_mode = OFF;
            PRAGMA temp_store = MEMORY;
        """)
        # PRAGMA locking_mode = EXCLUSIVE;
        self.con.commit()

        self.create_tables()

        # os.makedirs(log_path, exist_ok=True)
        # self.time_header= None
        # self.time_file = "performance_measures.csv"

        # self.set_performance_header()
    # def set_performance_header(self):

    def create_tables(self):
        if self.log_time:
            # Table for main simulation time measurement
            self.con.execute("""
                    CREATE TABLE  IF NOT EXISTS global_performance_measure(
                        scenario TEXT NOT NULL,
                        time_step INT NOT NULL,
                        total_sim_time REAL NOT NULL,
                        global_sim_preprocessing REAL,
                        global_batch_synchronization REAL,
                        global_visualization REAL,
                        --PRIMARY KEY(scenario, time_step)
                        PRIMARY KEY(scenario, time_step)
                       ) STRICT
                   """)


            # Table for batch simulation time measurement
            self.con.execute("""
                    CREATE TABLE  IF NOT EXISTS batch_performance_measure(
                        scenario TEXT NOT NULL,
                        batch TEXT NOT NULL,
                        time_step INT NOT NULL,
                        process_iteration_time REAL,
                        sim_step_time REAL NOT NULL,
                        agent_planning_time REAL NOT NULL,
                        sync_time_in REAL,
                        sync_time_out REAL,
                        -- PRIMARY KEY(scenario, batch, time_step)
                        PRIMARY KEY(scenario, batch, time_step)
                       ) STRICT
                   """)
            self.con.commit()

        # Table for general information (Scenarios
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS meta(
                scenario TEXT NOT NULL ,
                num_agents INT Not NULL,
                agent_ids ANY,
                original_planning_problem_id ANY,
                agent_batches ANY,
                duration_init REAL NOT NULL, 
                sim_duration REAL,
                post_duration REAL,
                simulation_config ANY NOT NULL,
                planner_config TEXT NOT NULL,
                PRIMARY KEY(scenario)
            ) STRICT
        """)

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS results(
                scenario TEXT NOT NULL ,
                agent_id INT NOT NULL,
                original_planning_problem INTEGER NOT NULL ,
                final_status INTEGER NOT NULL,
                last_timestep INTEGER NOT NULL ,
                message TEXT,
                agent_success TEXT NOT NULL,
                PRIMARY KEY(scenario, agent_id)
            ) STRICT
        """)


    def log_results(self,agents ):
        data = []
        for agent in agents:
            orig_pp = True if agent.id in self.original_planning_problem_id else False
            success = "success" if agent.status == AgentStatus.COMPLETED_SUCCESS else "failed"
            data.append([self.scenario, agent.id, orig_pp, agent.status, agent.agent_state.last_timestep, agent.agent_state.message, success])
        self.con.executemany("INSERT INTO results VALUES(?,?,?,?,?,?,?)", data)
        self.con.commit()

    def log_meta(self, agent_ids, original_planning_problem_id, batch_names, duration_init, config_sim, config_planner):
        self.original_planning_problem_id = original_planning_problem_id
        conf_sim = json.dumps(SimulationLogger._convert_config(config_sim))
        conf_plan = json.dumps(SimulationLogger._convert_config(config_planner))
        data = [self.scenario, len(agent_ids), json.dumps(agent_ids), json.dumps(original_planning_problem_id), json.dumps(batch_names), duration_init, None, None,conf_sim, conf_plan]
        self.con.execute("INSERT INTO meta VALUES(?,?,?,?,?,?,?,?,?,?)", data)
        self.con.commit()

    def update_meta(self, **kwargs):
        tmp = self.con.execute("select * from meta")
        cursor = self.con.cursor()
        cols = [tmp.description[i][0] for i in range(len(tmp.description))]
        cols2update = [key for key in kwargs.keys() if key in cols]
        cols2update = ""
        data = []
        for key, value in kwargs.items():
            if key in cols:
                cols2update += f"{key}= ?, "
                data.append(value)
        cols2update = cols2update[:-2]
        self.con.execute(f"UPDATE meta SET {cols2update} WHERE scenario = ?", data +[self.scenario])
        self.con.commit()

    def log_global_time(self, timestep, time_dict):
        data = [self.scenario, timestep,
                time_dict.pop("total_sim_step"),
                time_dict.pop("preprocessing"),
                time_dict.pop("time_sync") if "time_sync" in time_dict.keys() else None,
                time_dict.pop("time_visu") if "time_visu" in time_dict.keys() else None]
        self.con.execute("INSERT INTO global_performance_measure VALUES(?,?,?,?, ?,?)",data)
        self.con.commit()
        if len(time_dict) > 0:
            self.log_batch_time(timestep, time_dict)



    def log_batch_time(self,time_step, time_dict):
        data = []
        for batch_name, process_times in time_dict.items():
            data.append([self.scenario,batch_name,
                    time_step,
                    process_times["process_iteration_time"]  if "process_iteration_time" in process_times.keys() else None,
                    process_times["sim_step_time"],
                    process_times["agent_planning_time"],
                    process_times["sync_time_in"] if "sync_time_in" in process_times.keys() else None,
                    process_times["sync_time_out"] if "sync_time_out" in process_times.keys() else None,
                    ])
        self.con.executemany("INSERT INTO batch_performance_measure VALUES(?,?,?,?,?,?,?,?)", data)
        self.con.commit()


    # def log_meta(self, scenario, config):
    #     self.con.execute("INSERT INTO meta Values(?, ?)"(scenario.scenario_id, ))
    #     self.con.commit()


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
        os.makedirs(log_path, exist_ok=True)

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
