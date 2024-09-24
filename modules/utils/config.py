import os
import json


class Config:

    def __init__(self, config_name):
        with open(config_name, "r") as file:
            self.__dict__ = json.load(file)

    @staticmethod
    def default_config():
        return Config(os.path.join(os.path.dirname(__file__), "config.json"))
