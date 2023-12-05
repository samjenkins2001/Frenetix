import os
import sys
import csv
import shutil
from datetime import datetime
import traceback
import concurrent.futures
from cr_scenario_handler.simulation.simulation import Simulation
from cr_scenario_handler.utils.configuration_builder import ConfigurationBuilder
from cr_scenario_handler.utils.general import get_scenario_list
from omegaconf import OmegaConf, ListConfig, DictConfig
from cr_scenario_handler.evaluation.evaluation import evaluate_simulation


def run_simulation_wrapper(scenario_info):
    scenario_file, scenario_folder, mod_path, logs_path, use_cpp = scenario_info
    run_simulation(scenario_file, scenario_folder, mod_path, logs_path, use_cpp, start_multiagent=False)


def run_simulation(scenario_name, scenario_folder, mod_path, logs_path, use_cpp, start_multiagent=False, count=0):
    log_path = os.path.join(logs_path, scenario_name)
    config_sim = ConfigurationBuilder.build_sim_configuration(scenario_name, scenario_folder, mod_path)
    config_sim.simulation.use_multiagent = start_multiagent

    config_planner = ConfigurationBuilder.build_frenetplanner_configuration(scenario_name)
    config_planner.debug.use_cpp = use_cpp

    simulation = None
    evaluation = None

    try:
        simulation = Simulation(config_sim, config_planner)
        simulation.run_simulation()
        if config_sim.evaluation.evaluate_simulation:
            evaluation = evaluate_simulation(simulation)

    except Exception as e:
        error_traceback = traceback.format_exc()  # This gets the entire error traceback
        with open('logs/log_failures.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            current_time = datetime.now().strftime('%H:%M:%S')
            # Check if simulation is not None before trying to access current_timestep
            current_timestep = str(simulation.global_timestep) if simulation else "N/A"
            writer.writerow(["Scenario Name: " + log_path.split("/")[-1] + "\n" +
                             "Error time: " + str(current_time) + "\n" +
                             "In Scenario Timestep: " + current_timestep + "\n" +
                             "CODE ERROR: " + str(e) + error_traceback + "\n\n\n\n"])
            raise Exception
    return simulation, evaluation


def main():
    if sys.platform == "darwin":
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    mod_path = os.path.dirname(
        os.path.abspath(__file__)
    )
    stack_path = os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))
    logs_path = os.path.join(mod_path, "logs")

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
    scenario_name = "ZAM_Tjunction-1_42_T-1"  # do not add .xml format to the name
    scenario_folder = os.path.join(stack_path, "commonroad-scenarios", "scenarios")
    example_scenarios_list = os.path.join(mod_path, "example_scenarios", "scenario_list.csv")

    scenario_files = get_scenario_list(scenario_name, scenario_folder, evaluation_pipeline, example_scenarios_list,
                                       use_specific_scenario_list)

    # ***************************************************
    # Delete former logs & Create new score overview file
    # ***************************************************
    delete_former_logs = False
    if delete_former_logs:
        shutil.rmtree(logs_path, ignore_errors=True)
        os.makedirs(logs_path, exist_ok=True)
    if not os.path.exists(os.path.join(logs_path, "score_overview.csv")):
        with open(os.path.join(logs_path, "score_overview.csv"), 'a') as file:
            line = "scenario;agent;timestep;status;message\n"
            file.write(line)

    if evaluation_pipeline:
        if not start_multiagent:
            num_workers = 4  # or any number you choose based on your resources and requirements
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Create a list of tuples that will be passed to run_simulation_wrapper
                scenario_info_list = [(scenario_file, scenario_folder, mod_path, logs_path, use_cpp)
                                      for scenario_file in scenario_files]
                results = executor.map(run_simulation_wrapper, scenario_info_list)
        else:
            count = 0
            for scenario_file in scenario_files:
                run_simulation(scenario_file, scenario_folder, mod_path, logs_path, use_cpp,
                               start_multiagent, count)
                count += 1


    else:
        # If not in evaluation_pipeline mode, just run one scenario
        # config_eval
        simulation_result, evaluation_result = run_simulation(scenario_files[0], scenario_folder, mod_path, logs_path, use_cpp, start_multiagent)
        return simulation_result, evaluation_result


if __name__ == '__main__':
    simulation, evaluation = main()
