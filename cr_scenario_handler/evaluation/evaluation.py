from cr_scenario_handler.evaluation.crime_interface import CrimeInterface
from commonroad_crime.data_structure.configuration import CriMeConfiguration
from commonroad_crime.data_structure.crime_interface import CriMeInterface

def evaluate_simulation(agent_ids, scenario, config_sim):
    crime = CrimeInterface(config_sim, scenario)

    crime.set_agents(agent_ids)
    crime.evaluate()
    crime.plot_metrics()
