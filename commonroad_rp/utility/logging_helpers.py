import sys
import os
import logging
import json
from pathlib import Path


class DataLoggingCosts:
    # ----------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    def __init__(
        self, path_logs: str, header_only: bool = False
    ) -> None:
        """"""

        self.header = (
            "iteration;"
            "acceleration;"
            "jerk;"
            "jerk_lat;"
            "jerk_long;"
            "orientation;"
            "path_length;"
            "time_costs;"
            "inverse_duration;"
            "velocity;"
            "dist_obj;"
            "prediction;"
        )
        file_name = "costs_logs.csv"
        if header_only:
            return
        self.iteration = 0
        # Create directories
        if not os.path.exists(path_logs):
            os.makedirs(path_logs)
        self.__log_path = os.path.join(path_logs, file_name)
        Path(os.path.dirname(self.__log_path)).mkdir(parents=True, exist_ok=True)

        # write header to logging file
        with open(self.__log_path, "w+") as fh:
            fh.write(self.header)

    # ----------------------------------------------------------------------------------------------------------
    # CLASS METHODS --------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------

    def get_headers(self):
        return self.header

    def log_data(
        self,
        costs: list
    ) -> None:
        """log_data _summary_
        """
        with open(self.__log_path, "a") as fh:
            fh.write(
                "\n"
                + str(self.iteration)
                + ";"
                + json.dumps(str(costs[0]))
                + ";"
                + json.dumps(str(costs[1]))
                + ";"
                + json.dumps(str(costs[2]))
                + ";"
                + json.dumps(str(costs[3]))
                + ";"
                + json.dumps(str(costs[4]))
                + ";"
                + json.dumps(str(costs[5]))
                + ";"
                + json.dumps(str(costs[6]))
                + ";"
                + json.dumps(str(costs[7]))
                + ";"
                + json.dumps(str(costs[8]))
                + ";"
                + json.dumps(str(costs[9]))
                + ";"
                + json.dumps(str(costs[10]))
            )
