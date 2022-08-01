from commonroad_rp.run_planner import run_planner
from commonroad_rp.configuration_builder import ConfigurationBuilder

if __name__ == '__main__':

    scenario_name = "DEU_Flensburg-24_1_T-1"

    config = ConfigurationBuilder.build_configuration(scenario_name)
    log_path = "./logs/"+scenario_name
    cost_function_path = "configurations/cost_weights.yaml"

    run_planner(config, log_path, cost_function_path)
