"""
Module for variables that are used throughout the project.

Inludes:
    - ROOT_PATH: Absolute path to root of project
    - DATA_PATH: Absolute path to data directory
    - MODELS_PATH: Absolute path to models directory
    - TESTS_PATH: Absolute path to tests directory
"""

import os
import pathlib

# Paths
ROOT_PATH = pathlib.Path(__file__).parent.parent.absolute().as_posix()
DATA_PATH = os.path.join(ROOT_PATH, "data")
MODELS_PATH = os.path.join(ROOT_PATH, "models")
TESTS_PATH = os.path.join(ROOT_PATH, "tests")
