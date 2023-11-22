from abc import abstractmethod, ABC
from commonroad_crime.data_structure.crime_interface import CriMeInterface
from typing import List, Type

from commonroad_crime.data_structure.base import CriMeBase
from commonroad_crime.data_structure.configuration import CriMeConfiguration
import commonroad_crime.measure as measures
import matplotlib
import matplotlib.pyplot as plt
import math
from commonroad_crime.data_structure.type import TypeMonotone
import numpy as np
import pandas as pd
import commonroad_crime.utility.logger as utils_log
from multiprocessing import Pool
import concurrent.futures
import copy as copy
from itertools import repeat

class CrimeInterface(ABC):
    def __init__(self, config_sim, scenario):
        self.id = scenario.scenario_id
        self.config = config_sim.evaluation
        self.scenario = scenario
        self.start_min = None
        self.end_max = None

        self.config_measures = CriMeConfiguration()
        self.config_measures.update(sce=scenario)

        self._measure_names = None
        self._measures = None
        self.set_measures(self.config)

        self._agent_ids = None

        self._df_criticality = None
        self._measure_evaluators = []
        # self._crime_interface = None
        return

    @property
    def agent_ids(self):
        return self._agent_ids

    @property
    def criticality_dataframe(self):
        return self._df_criticality

    # @property
    # def measures(self):
    #     return self._measures

    def set_measures(self, config):
        self._measure_names = [metric for metric, used
                               in config.criticality_metrics.items()
                               if hasattr(measures, metric) and used]

        self._measures =  [getattr(measures, metric) for metric in self._measure_names]

    def set_agents(self, agent_id):
        self.start_min = min(self.scenario.obstacle_by_id(agent).initial_state.time_step for agent in agent_id)
        self.end_max = max(self.scenario.obstacle_by_id(agent).prediction.final_time_step for agent in agent_id)

        self._agent_ids = agent_id
        self._df_criticality = pd.DataFrame.from_dict({(i,j): dict.fromkeys(self._measure_names, np.inf)
                                                       for i in agent_id
                                                       for j in range(self.start_min, self.end_max+1)},
                                                      orient='index')
        #TODO fill df and use df for visualization!
        # (
        #
        # dict.fromkeys(agent_id, dict.fromkeys(range(self.start_min, self.end_max), dict.fromkeys(self._measure_names, np.inf))))
        return

    def evaluate(self):

        # t_start = min(self.scenario.obstacle_by_id(i).initial_state.time_step for i in self.agent_ids)
        # t_end = max(self.scenario.obstacle_by_id(i).prediction.final_time_step for i in self.agent_ids)
        for agent_id in self.agent_ids:
            self.evaluate_agent(agent_id)



    def evaluate_agent(self, agent_id):
        self.config_measures.update(ego_id=agent_id)
        t_start = self.scenario.obstacle_by_id(agent_id).initial_state.time_step
        t_end = self.scenario.obstacle_by_id(agent_id).prediction.final_time_step
        for t in range(t_start, t_end + 1):
            results = self.evaluate_timestep(agent_id, t, self._measures, self.config_measures)
            self._df_criticality.loc[agent_id, t] = results
            # self._criticality_dict[agent_id].update(results)

        # with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
        #     args_list = [(agent_id, t, self._measures, self.config_measures) for t in range(t_start, t_end + 1)]
        #     # for result in pool.map(evaluation_wrapper, args_list):
        #     #     print(result)
        #     futures = [pool.submit(evaluate_timestep, agent_id, t, copy.deepcopy(self._measures), copy.deepcopy(self.config_measures)) for t in range(t_start, t_end + 1)]
        #     for future in futures:
        #         print(future.result())
        print("done")

    def evaluate_timestep(self, agent_id, time_step, used_measures, config_measures):
        results = pd.Series(0, index=self._measure_names)
        # results = {time_step: {}}
        # if time_step not in self._criticality_dict[agent_id] :
        #     self._criticality_dict[agent_id][time_step] = {}
        for name, measure in zip(self._measure_names, self._measures):
            # if measure not in self.measures:
            #     self.measures.append(measure)
            # print(measure.measure_name.value)
            m_evaluator = measure(self.config_measures)
            results[name] = m_evaluator.compute_criticality(
                time_step, verbose=False)
            # results[time_step][name] = m_evaluator.compute_criticality(
            #     time_step, verbose=False)
            # # if measure.measure_name.value not in self._criticality_dict[agent_id][time_step]:
            # self._measure_evaluators.append(m_evaluator)
        # print(results)
        return results

    def get_metric_for_agent(self, agent_id, metric):
        return [self._criticality_dict[agent_id][t][metric] for t in self._criticality_dict[agent_id].keys()]

    def plot_metrics(self, nr_per_row=2, flag_latex=True):
        # if flag_latex:
        #     # use Latex font
        #     FONTSIZE = 28
        #     plt.rcParams["text.latex.preamble"] = r"\usepackage{lmodern}"
        #     pgf_with_latex = {  # setup matplotlib to use latex for output
        #         "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
        #         "text.usetex": True,  # use LaTeX to write all text
        #         "font.family": "lmodern",
        #         # blank entries should cause plots
        #         "font.sans-serif": [],  # ['Avant Garde'],              # to inherit fonts from the document
        #         # 'text.latex.unicode': True,
        #         "font.monospace": [],
        #         "axes.labelsize": FONTSIZE,  # LaTeX default is 10pt font.
        #         "font.size": FONTSIZE - 10,
        #         "legend.fontsize": FONTSIZE,  # Make the legend/label fonts
        #         "xtick.labelsize": FONTSIZE,  # a little smaller
        #         "ytick.labelsize": FONTSIZE,
        #         "pgf.preamble": r"\usepackage[utf8x]{inputenc}"
        #                         + r"\usepackage[T1]{fontenc}"
        #                         + r"\usepackage[detect-all,locale=DE]{siunitx}",
        #     }
        #     matplotlib.rcParams.update(pgf_with_latex)
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
                if measure.monotone == TypeMonotone.NEG:
                    ax.invert_yaxis()
                count_column += 1
                if count_column > nr_per_row - 1:
                    count_column = 0
                    count_row += 1
            fig.suptitle(self.id)
            plt.show()



# def evaluation_wrapper(args):
#     agent_id, time_step, used_measures, config_measures = args
#     return evaluate_timestep(agent_id, time_step, used_measures, config_measures)

