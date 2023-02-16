import os
import sys
from commonroad_rp.run_planner import run_planner
from commonroad_rp.configuration_builder import ConfigurationBuilder

mod_path = os.path.dirname(
    os.path.abspath(__file__)
)
sys.path.append(mod_path)
stack_path = os.path.dirname(
    os.path.abspath(__file__)
)

if __name__ == '__main__':

    scenario_name = "ZAM_Tjunction-1_109_T-1"
    scenario_path = os.path.join(mod_path, "example_scenarios", scenario_name)
    # scenario_path = os.path.join(stack_path, "commonroad-scenarios", "scenarios", "recorded", "scenario-factory", scenario_name)
    config = ConfigurationBuilder.build_configuration(scenario_path+".xml")
    log_path = "./logs/"+scenario_path.split("/")[-1]
    cost_function_path = "configurations/cost_weights.yaml"

    run_planner(config, log_path, mod_path)
