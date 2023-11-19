import os
import pathlib

# Paths
ROOT_PATH = pathlib.Path(__file__).parent.parent.absolute().as_posix()
DATA_PATH = os.path.join(ROOT_PATH, "data")
MODELS_PATH = os.path.join(ROOT_PATH, "models")
TESTS_PATH = os.path.join(ROOT_PATH, "tests")
