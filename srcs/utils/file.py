# -- coding: utf-8 --
import os

from utils.env import Env

def prepare_directory(dir_name):
    file_dir = os.path.join(Env.get_project_dir(), dir_name)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    return file_dir