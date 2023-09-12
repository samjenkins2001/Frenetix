import os
import random
import sys
import csv
from os import listdir
from os.path import isfile, join
from commonroad_rp.run_planner import run_planner
from cr_scenario_handler.simulation.simulation import Simulation
from cr_scenario_handler.utils.configuration_builder import ConfigurationBuilder
from cr_scenario_handler.utils.general import read_scenario_list


def get_scenario_list(scenario_folder, example_scenarios_list, use_specific_scenario_list):
    if not use_specific_scenario_list:
        scenario_files = [f.split(".")[-2] for f in listdir(scenario_folder) if isfile(join(scenario_folder, f))]
        random.shuffle(scenario_files)
    else:
        scenario_files = read_scenario_list(example_scenarios_list)
        random.shuffle(scenario_files)
    return scenario_files


def main():
    if sys.platform == "darwin":
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    mod_path = os.path.dirname(
        os.path.abspath(__file__)
    )
    sys.path.append(mod_path)
    stack_path = os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))

    start_multiagent = False
    use_cpp = True
    evaluation_pipeline = False
    use_specific_scenario_list = False

    scenario_name = "DEU_Flensburg-12_1_T-1"  # do not add .xml format to the name
    scenario_folder = os.path.join(stack_path, "commonroad-scenarios", "scenarios")
    example_scenarios_list = os.path.join(mod_path, "example_scenarios", "scenario_list.csv")

    scenario_files = get_scenario_list(scenario_folder, example_scenarios_list, use_specific_scenario_list)

    number_of_runs = len(scenario_files) if evaluation_pipeline else 1

    for runs in range(0, number_of_runs):

        scenario_path = os.path.join(scenario_folder, scenario_files[runs]) if evaluation_pipeline \
            else os.path.join(scenario_folder, scenario_name)

        config = ConfigurationBuilder.build_configuration(scenario_path + ".xml")
        log_path = "./logs/"+scenario_path.split("/")[-1]

        try:
            if not start_multiagent:
                run_planner(config, log_path, mod_path, use_cpp)
            else:
                # Works only with wale-net. Ground Truth Prediction not possible!
                simulation = Simulation(config, log_path, mod_path)
                simulation.run_simulation()
        except Exception as e:
            with open('logs/log_failures.csv', 'a', newline='') as f:
                csv.writer(f).writerow([log_path.split("/")[-1] + " --> CODE ERROR: " + str(e)])


if __name__ == '__main__':
    main()

