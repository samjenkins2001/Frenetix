import os
import random
import sys
import csv
import traceback
import concurrent.futures
from os import listdir
from os.path import isfile, join
from frenetix_motion_planner.run_planner import run_planner
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


def run_simulation_wrapper(scenario_info):
    scenario_file, mod_path, scenario_folder, start_multiagent, use_cpp = scenario_info
    scenario_path = os.path.join(scenario_folder, scenario_file)
    log_path = "./logs/" + scenario_file.split("/")[-1]
    run_simulation(scenario_path, mod_path, log_path, start_multiagent, use_cpp)


def run_simulation(scenario_path, mod_path, log_path, start_multiagent, use_cpp):
    config = ConfigurationBuilder.build_configuration(scenario_path + ".xml", dir_config_default='defaults')
    simulation = None
    try:
        if not start_multiagent:
            if not use_cpp:
                run_planner(config, log_path, mod_path, use_cpp)
            else:
                simulation = Simulation(config, log_path, mod_path)
                simulation.run_simulation()
        else:
            # Works only with wale-net. Ground Truth Prediction not possible!
            simulation = Simulation(config, log_path, mod_path)
            simulation.run_simulation()
    except Exception as e:
        error_traceback = traceback.format_exc()  # This gets the entire error traceback
        with open('logs/log_failures.csv', 'a', newline='') as f:
            csv.writer(f).writerow([log_path.split("/")[-1], "In Timestep: ", str(simulation.current_timestep),
                                    " --> CODE ERROR: ", str(e), error_traceback, "\n\n"])
        print(error_traceback)


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

    # *************************
    # Set Python or C++ Planner
    # *************************
    use_cpp = True

    # ****************************************************************************************************
    # Start Multiagent Problem. Does not work for every Scenario and Setting. Try "ZAM_Tjunction-1_42_T-1"
    # ****************************************************************************************************
    start_multiagent = False

    # *********************************************************
    # Link a Scenario Folder & Start many Scenarios to evaluate
    # *********************************************************
    evaluation_pipeline = False

    # ******************************************************************************************************
    # Setup a specific scenario list to evaluate. The scenarios in the list have to be in the example folder
    # ******************************************************************************************************
    use_specific_scenario_list = False

    # **********************************************************************
    # If the previous are set to "False", please specify a specific scenario
    # **********************************************************************
    scenario_name = "BEL_Putte-7_5_T-1"  # do not add .xml format to the name
    scenario_folder = os.path.join(stack_path, "commonroad-scenarios", "scenarios")  # Change to CommonRoad scenarios folder if needed.
    example_scenarios_list = os.path.join(mod_path, "example_scenarios", "scenario_list.csv")

    scenario_files = get_scenario_list(scenario_folder, example_scenarios_list, use_specific_scenario_list)

    if evaluation_pipeline and not start_multiagent:
        num_workers = 6  # or any number you choose based on your resources and requirements
        with open(os.path.join(mod_path, "logs", "score_overview.csv"), 'a') as file:
            line = "scenario;timestep;result\n"
            file.write(line)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Create a list of tuples that will be passed to run_simulation_wrapper
            scenario_info_list = [(scenario_file, mod_path, scenario_folder, start_multiagent, use_cpp)
                                  for scenario_file in scenario_files]
            results = executor.map(run_simulation_wrapper, scenario_info_list)

    else:
        # If not in evaluation_pipeline mode, just run one scenario
        scenario_path = os.path.join(scenario_folder, scenario_name)
        log_path = "./logs/" + scenario_name
        run_simulation(scenario_path, mod_path, log_path, start_multiagent, use_cpp)


if __name__ == '__main__':
    main()

