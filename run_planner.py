import os
import sys
from commonroad_rp.run_planner import run_planner
from commonroad_rp.configuration_builder import ConfigurationBuilder

mopl_path = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))
sys.path.append(mopl_path)


if __name__ == '__main__':

    scenario_name = "ZAM_Tjunction-1_81_T-1"
    # scenario_path = os.path.join(mopl_path, "commonroad-reactive-planner", "example_scenarios", scenario_name)
    scenario_path = os.path.join(mopl_path, "commonroad-scenarios", "scenarios", "recorded", "hand-crafted", scenario_name)
    config = ConfigurationBuilder.build_configuration(scenario_path+".xml")
    log_path = "./logs/"+scenario_path.split("/")[-1]
    cost_function_path = "configurations/cost_weights.yaml"

    run_planner(config, log_path)
