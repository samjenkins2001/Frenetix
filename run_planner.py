from commonroad_rp.run_planner import run_planner
from commonroad_rp.configuration import build_configuration

if __name__ == '__main__':
    base_dir = "example_scenarios"
    #filename = "USA_Lanker-2_6_T-1"
    scenario_name = "USA_Lanker-2_6_T-1"

    config = build_configuration(scenario_name)
    log_path = "./logs/"+scenario_name
    cost_function_path = "configurations/cost_weights.yaml"

    run_planner(base_dir, scenario_name, config, log_path, cost_function_path)
