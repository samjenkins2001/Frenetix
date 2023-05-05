import os
import sys
from commonroad_rp.run_planner import run_planner
from multiagent.simulation import Simulation
from commonroad_rp.configuration_builder import ConfigurationBuilder

if sys.platform == "darwin":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

mod_path = os.path.dirname(
    os.path.abspath(__file__)
)
sys.path.append(mod_path)
stack_path = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))

if __name__ == '__main__':

    scenario_name = "ZAM_Tjunction-1_294_T-1"
    scenario_path = os.path.join(stack_path, "commonroad-scenarios", "scenarios", scenario_name)

    config = ConfigurationBuilder.build_configuration(scenario_path+".xml")
    log_path = "./logs/"+scenario_path.split("/")[-1]

    simulation = Simulation(config, log_path, mod_path)
    simulation.run_simulation()
