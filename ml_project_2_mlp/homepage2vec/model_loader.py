"""
Module to load the homepage2vec model into the project.

Includes:
    - get_model_path: Function to get the path to the homepage2vec model.
"""

import os

from ..conf import MODELS_PATH


def get_model_path() -> tuple[str, str]:
    """
    Returns the path to the homepage2vec model.

    Returns:
        model_home: Path to the folder containing the homepage2vec model.
        model_folder: Path to the homepage2vec model.
    """
    model_home = MODELS_PATH
    model_folder = os.path.join(model_home, "homepage2vec")

    if not os.path.exists(model_folder):
        # TODO: Add logic to download model from GDrive
        pass

    return model_home, model_folder
