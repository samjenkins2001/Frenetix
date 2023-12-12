import math
from abc import ABC

# import commonroad_crime.measure as measures
import matplotlib.pyplot as plt
import pandas as pd

from cr_scenario_handler.evaluation.metrics import Measures


def evaluate_agent(agent_id, scenario, reference_paths, metrics, msg_logger):
    # self.config_measures.update(ego_id=agent_id)
    t_start = scenario.obstacle_by_id(agent_id).initial_state.time_step
    t_end = scenario.obstacle_by_id(agent_id).prediction.final_time_step

    results = pd.DataFrame(None, index=list(range(t_start,t_end+1)), columns=metrics)
    # for t in range(t_start, t_end + 1):
    measures = Measures(agent_id, scenario, t_start, t_end, reference_paths, msg_logger)
    for measure in metrics:
        value = getattr(measures, measure)()
        results[measure] = value
        # results = evaluate_timestep(agent_id, t, scenario, measures)
        # self._df_criticality.loc[agent_id, t] = results
    return results







class Evaluator(ABC):
    def __init__(self, config_sim, scenario, msg_logger, sim_logger, agent_ids=None, agents=None):
        self.sim_logger = sim_logger
        self.msg_logger = msg_logger
        self.id = scenario.scenario_id
        self.config = config_sim.evaluation
        self.scenario = scenario

        self.start_min = None
        self.end_max = None

        # self.config_measures = CriMeConfiguration()
        # self.config_measures.update(sce=scenario)

        # self._measure_names = None
        self.measures = dict()
        self.set_measures(self.config)

        self._agents = None
        self._agent_ids = None
        self._df_criticality = None
        self.reference_paths = dict()
        self.set_agents(agent_ids, agents)


        # self._measure_evaluators = []
        # self._crime_interface = None
        return

    @property
    def agent_ids(self):
        return self._agent_ids

    @property
    def agents(self):
        return self._agents

    @property
    def criticality_dataframe(self):
        return self._df_criticality

    # @property
    # def measures(self):
    #     return self._measures

    def set_measures(self, config):
        # self._measure_names = [metric for metric, used
        #                        in config.criticality_metrics.items()
        #                        if hasattr(measures, metric) and used]

        self.measures = [metric  for metric, used
                               in config.criticality_metrics.items()
                               if hasattr(Measures, metric) and used]

    def set_agents(self, agent_id=None, agents=None):
        if agent_id and agents:
            ids = [i.id for i in agents]
            if not all([i in ids for i in agent_id]) and not all([i in agent_id for i in ids]):
                raise ValueError("provided agents_ids do not match with agents!")
        elif agents:
            agent_id = [i.id for i in agents]
            self.reference_paths= {i.id: i.reference_paths for i in agents}
        elif not agent_id and not agents:
            agent_id = [obs.id for obs in self.scenario.dynamic_obstacles]

        self.start_min = min(self.scenario.obstacle_by_id(agent).initial_state.time_step for agent in agent_id)
        self.end_max = max(self.scenario.obstacle_by_id(agent).prediction.final_time_step for agent in agent_id)

        self._agent_ids = agent_id
        self._agents = agents
        # self._df_criticality = pd.DataFrame.from_dict({(i,j): dict.fromkeys(self.measures, None)
        #                                                for i in agent_id
        #                                                for j in range(self.start_min, self.end_max+1)},
        #                                               orient='index')



    def evaluate(self):

        for agent_id in self.agent_ids:
            self.msg_logger.critical(f"Evaluate agent {agent_id}")
            self.msg_logger.info(f"metrics: {self.measures}")
            ref_path = self.reference_paths[agent_id] if agent_id in self.reference_paths else None
            # cosy =  self.coordinatie_system[agent_id] if agent_id in self.coordinatie_system else None
            agent_results = evaluate_agent(agent_id, self.scenario, ref_path, self.measures, self.msg_logger)
            if self._df_criticality is None:
                self._df_criticality = agent_results.set_index([[agent_id] * len(agent_results), agent_results.index])
            else:
                df_to_append = agent_results.set_index([[agent_id] * len(agent_results), agent_results.index])
                self._df_criticality = pd.concat([self._df_criticality, df_to_append])
      # with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
        #     args_list = [(agent_id, t, self._measures, self.config_measures) for t in range(t_start, t_end + 1)]
        #     # for result in pool.map(evaluation_wrapper, args_list):
        #     #     print(result)
        #     futures = [pool.submit(evaluate_timestep, agent_id, t, copy.deepcopy(self._measures), copy.deepcopy(self.config_measures)) for t in range(t_start, t_end + 1)]
        #     for future in futures:
    #     #         print(future.result())



    def plot_metrics(self, nr_per_row=2, flag_latex=True):
        if (self._measures is not None and self.start_min is not None and self.end_max is not None):
            nr_metrics = len(self._measures)
            if nr_metrics > nr_per_row:
                nr_column = nr_per_row
                nr_row = math.ceil(nr_metrics / nr_column)
            else:
                nr_column = nr_metrics
                nr_row = 1
            fig, axs = plt.subplots(
                nr_row, nr_column, figsize=(7.5 * nr_column, 5 * nr_row)
            )
            count_row, count_column = 0, 0
            df = self._df_criticality.unstack(level=0)
            for name, measure in zip(self._measure_names, self._measures):
                if nr_metrics == 1:
                    ax = axs
                elif nr_row == 1:
                    ax = axs[count_column]
                else:
                    ax = axs[count_row, count_column]
                df[name].plot(ax=ax)

                ax.set_title(name)
                ax.set_xlabel("Time step")
                ax.set_ylabel("Value")
                # if measure.monotone == TypeMonotone.NEG:
                #     ax.invert_yaxis()
                count_column += 1
                if count_column > nr_per_row - 1:
                    count_column = 0
                    count_row += 1
            fig.suptitle(self.id)
            plt.show()




def evaluate_simulation(simulation):
    config = simulation.config
    if not config.evaluation.evaluate_simulation:
        return

    agents = simulation.agents
    agent_ids = simulation.agent_id_list
    scenario = simulation.scenario
    sim_logger = simulation.sim_logger
    msg_logger = simulation.msg_logger

    evaluation = Evaluator(config, scenario, msg_logger, sim_logger, agent_ids, agents)
    evaluation.evaluate()
    # eval.plot_metrics()
    return evaluation

