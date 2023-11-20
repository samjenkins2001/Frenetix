import os
import sys
import csv
import traceback
import concurrent.futures
from cr_scenario_handler.simulation.simulation import Simulation
from cr_scenario_handler.utils.configuration_builder import ConfigurationBuilder
from cr_scenario_handler.utils.general import get_scenario_list


def run_simulation_wrapper(scenario_info):
    scenario_file, scenario_folder, mod_path, use_cpp = scenario_info
    run_simulation(scenario_file, scenario_folder, mod_path, use_cpp)


def run_simulation(scenario_name, scenario_folder, mod_path, use_cpp):
    log_path = "./logs/" + scenario_name
    config_sim = ConfigurationBuilder.build_sim_configuration(scenario_name, scenario_folder, mod_path)
    config_planner = ConfigurationBuilder.build_frenetplanner_configuration(scenario_name)
    config_planner.debug.use_cpp = use_cpp

    simulation = None
    try:
        simulation = Simulation(config_sim, config_planner)
        simulation.run_simulation()
    except Exception as e:
        error_traceback = traceback.format_exc()  # This gets the entire error traceback
        with open('logs/log_failures.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            # Check if simulation is not None before trying to access current_timestep
            current_timestep = str(simulation.global_timestep) if simulation else "N/A"
            writer.writerow([log_path.split("/")[-1], "In Timestep: ", current_timestep,
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
    if not use_cpp and start_multiagent:
        raise IOError("Starting Multiagent with python is strongly not recommended!")

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
    scenario_name = "ZAM_Tjunction-1_180_T-1"  # do not add .xml format to the name
    scenario_folder = os.path.join(stack_path, "commonroad-scenarios", "scenarios")
    example_scenarios_list = os.path.join(mod_path, "example_scenarios", "scenario_list.csv")

    scenario_files = get_scenario_list(scenario_name, scenario_folder, evaluation_pipeline, example_scenarios_list,
                                       use_specific_scenario_list)

    if evaluation_pipeline and not start_multiagent:
        num_workers = 4  # or any number you choose based on your resources and requirements
        with open(os.path.join(mod_path, "logs", "score_overview.csv"), 'a') as file:
            line = "scenario;timestep;status;message\n"
            file.write(line)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Create a list of tuples that will be passed to run_simulation_wrapper
            scenario_info_list = [(scenario_file, scenario_folder, mod_path, use_cpp)
                                  for scenario_file in scenario_files]
            results = executor.map(run_simulation_wrapper, scenario_info_list)

    else:
        # If not in evaluation_pipeline mode, just run one scenario
        run_simulation(scenario_files[0], scenario_folder, mod_path, use_cpp)


if __name__ == '__main__':
    main()

