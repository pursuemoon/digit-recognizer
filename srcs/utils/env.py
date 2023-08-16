# -- coding: utf-8 --

import os

project_dir = None

class Env(object):

    LOG_DIR = "logs"
    MODEL_DIR = "models"

    @staticmethod
    def get_project_dir():
        global project_dir
        if not project_dir:
            project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return project_dir