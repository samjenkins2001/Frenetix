from commonroad_rp.run_planner import run_planner
from commonroad_rp.configuration_builder import ConfigurationBuilder

if __name__ == '__main__':

    scenario_name = "/media/sf_6.Semester/Planner/commonroad-reactive-planner/example_scenarios/DEU_Flensburg-24_1_T-1"

    config = ConfigurationBuilder.build_configuration(scenario_name+".xml")
    log_path = "./logs/"+scenario_name.split("/")[-1]
    cost_function_path = "configurations/cost_weights.yaml"

    run_planner(config, log_path, cost_function_path)
