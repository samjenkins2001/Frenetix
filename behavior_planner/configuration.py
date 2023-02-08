from typing import Union
from omegaconf import ListConfig, DictConfig


class Configuration:
    """
    Main Configuration class holding all planner-relevant configurations
    """
    def __init__(self, config: Union[ListConfig, DictConfig]):
        # initialize subclasses
        self.mission: MissionConfiguration = MissionConfiguration(config.mission)
        self.mode: ModeConfiguration = ModeConfiguration(config.mode)


class MissionConfiguration:
    """Class to store all planning configurations"""
    def __init__(self, config: Union[ListConfig, DictConfig]):
        self.priorityright = config.priorityright


class ModeConfiguration:
    """Class to store all prediction configurations"""
    def __init__(self, config: Union[ListConfig, DictConfig]):
        self.overtaking = config.overtaking