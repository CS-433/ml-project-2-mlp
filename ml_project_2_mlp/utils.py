"""
Module containing utility functions.

Functions:
    check_import: Checks if the module can be imported.
    load_curlie_data: Loads the original processed Curlie data that was used
    load_crowdsourced_data: Loads the crowdsourced data from the data directory.
    load_homepage2vec: Loads the pre-trained Homepage2Vec from the model directory.
"""

import gzip
import json
import logging
import os
import shutil
import warnings
import zipfile
from typing import List, Sequence

import gdown
import hydra
import pandas as pd
import rich
import rich.syntax
import rich.tree
import torch
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

log = logging.Logger(__name__)


def check_import() -> bool:
    """Checks if the module can be imported."""
    return True


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Rich config printing

    Args:
        cfg: A DictConfig object containing the config tree.
    """
    # Return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # Disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # Print config with rich
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        print_config_tree(cfg, resolve=True)


def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "data",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """
    Prints the contents of a DictConfig as a tree structure using the Rich library.

    Args:
        cfg: A DictConfig composed by Hydra.
        print_order: Determines in what order config components are printed. Default is ``("data", "model",
        "callbacks", "logger", "trainer", "paths", "extras")``.
        resolve: Whether to resolve reference fields of DictConfig. Default is ``False``.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # Add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else log.warning(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        )

    # Add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # Generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)


def log_hyperparameters(setup_dict: dict) -> None:
    """
    Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters (trainable, non-trainable, total)

    Args:
        setup_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
        - `"callbacks"`: A list of Lightning callbacks.
        - `"extras"`: A list of extra objects to save.
    """
    hparams = {}

    cfg = OmegaConf.to_container(setup_dict["cfg"])
    model = setup_dict["model"]
    trainer = setup_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    # Save information on run
    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["seed"] = cfg.get("seed")

    # Save model, data, trainer and callback configs
    hparams["model"] = cfg["model"]
    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]
    hparams["callbacks"] = cfg.get("callbacks")

    # Additionally: Save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # Log hyperparameters in all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """
    Instantiates callbacks for lightning.Trainer from Hydra configuration.

    Args:
        callback_cfg: Hydra configurations for callbacks

    Returns:
        A list of instantiated callbacks.
    """
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """
    Instantiates loggers from config.

    Args:
        logger_cfg: A DictConfig object containing logger configurations.

    Returns:
        A list of instantiated loggers.
    """
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


def load_from_gdrive(dir_path: str, gdrive_url: str, expected_files: list[str]):
    """
    Loads folder of data/ model from Google Drive if not present in the specified
    directory. Automatically extracts compressed files and reads them in based on the
    file extension.

    Args:
        dir_path: Path to the directory containing the data.
        gdrive_url: URL to the data on Google Drive.
        expected_files: List of expected files in the directory.

    Returns:
        Tuple of data from the files (in same order as expected_files).
    """
    # Create directory if not present
    os.makedirs(dir_path, exist_ok=True)

    # Download if not present
    download_if_not_present(
        dir_path=dir_path, expected_files=expected_files, gdrive_url=gdrive_url
    )

    # Load data
    data = load(dir_path=dir_path, expected_files=expected_files)

    return data


def load(dir_path: str, expected_files: list[str]):
    # Load data
    expected_file_paths = list(map(lambda x: os.path.join(dir_path, x), expected_files))
    data = _extract_and_read_files(expected_file_paths)
    return data


def download_if_not_present(
    dir_path: str, expected_files: list[str], gdrive_url: str
) -> None:
    """
    Downloads the data/ model from Google Drive if not present in
    the specified directory. Not present means that either the
    directory does not exist or one of the expected files is missing.
    Otherwise, downloads the data/ model to the specified directory.

    Args:
        dir_path: Path to the directory containing the files
        expected_files: List of expected files in the directory.
        gdrive_url: Google Drive URL to the zipped files.
    """
    # Check if directory exists
    dir_exists = os.path.exists(dir_path)
    if not dir_exists:
        os.makedirs(dir_path, exist_ok=True)
        log.info(f"{dir_path} doesn't exist. Downloading from Google Drive...")
        _download_from_gdrive(gdrive_url=gdrive_url, path_dir=dir_path)

    # Check if all expected files are present (if file that matches the first part of the file name exists)
    files_in_dir = os.listdir(dir_path)
    all_files_exist = all(
        map(
            lambda x: any(map(lambda y: y.startswith(x.split(".")[0]), files_in_dir)),
            expected_files,
        )
    )

    if not all_files_exist:
        log.info(f"{dir_path} doesn't exist. Downloading from Google Drive...")
        _download_from_gdrive(gdrive_url=gdrive_url, path_dir=dir_path)
    else:
        log.debug(f"Data found at {dir_path}.")


def _download_from_gdrive(gdrive_url: str, path_dir: str) -> None:
    """
    Downloads a zipped file from Google Drive and extracts it
    into the specified path.

    Args:
        grdrive_url: Google Drive URL to the zipped file.
        path_dir: Directory where the data will be extracted.

    Returns:
        None
    """
    # Create temporary name for zip file
    file_path = path_dir + ".zip"

    # Download zipped folder from Google Drive
    log.debug(f"Downloading data from Google Drive to {file_path}...")
    gdown.download(gdrive_url, file_path)

    # Extract zipped folder depending on compression type
    try:
        _extract_zip(file_path)
    except Exception as e:
        log.error(f"Could not extract zip file: {e}")

    # Remove temporary zip file
    os.remove(file_path)


def _extract_and_read_files(file_paths: list[str]) -> tuple:
    """
    Reads a list of files based on the file extension. If the file is
    present per exact match it tries to find a compressed version of
    the file and extracts it if found.

    Args:
        file_paths: List of paths to the files.

    Returns:
        data: Tuple containing the data from the files.
    """
    data = []
    for file_path in file_paths:
        # Searches for compressed file if not exact match
        if not os.path.exists(file_path):
            log.debug(f"Could not find {file_path}. Searching for compressed file...")
            path_parts = os.path.split(file_path)
            path, _ = os.path.join(*path_parts[:-1]), path_parts[1]
            possible_files = map(
                lambda x: os.path.join(path, x), os.listdir(os.path.dirname(file_path))
            )
            compressed_file_path = list(
                filter(lambda x: x.startswith(file_path), possible_files)
            )[0]
            file_path = _extract_file(compressed_file_path)

        # Read file
        file_content = _read_file(file_path)
        data.append(file_content)

    return tuple(data)


def _extract_file(file_path: str, remove: bool = True) -> None:
    """
    Extracts a compressed file in the current directory if it is
    compressed according to the available compression types. Otherwise
    does nothing.

    Args:
        file_path (str): Path to the compressed file.
        remove (bool): Whether to remove the compressed file after extraction.

    Returns:
        file_path (str): Path to the extracted file.
    """
    path_parts = file_path.split("/")
    path, file_name = "/" + os.path.join(*path_parts[:-1]), path_parts[-1]

    file_parts = file_name.split(".")
    file_name, file_extension = ".".join(file_parts[:-1]), file_parts[-1]

    match file_extension:
        case "zip":
            _extract_zip(file_path)
        case "gz":
            _extract_gzip(file_path)
        case _:
            return file_path
    if remove:
        os.remove(file_path)

    return os.path.join(path, file_name)


def _extract_zip(file_path: str) -> None:
    """
    Extracts a zipped file in the current directory.

    Args:
        file_path: Path to the zipped file.

    Returns:
        None
    """
    dir_path = os.path.dirname(file_path)
    log.debug(f"Extracting data from {file_path}")
    with zipfile.ZipFile(file_path, "r") as f:
        f.extractall(dir_path)


def _extract_gzip(file_path: str) -> None:
    """
    Extracts a gzipped file in the current directory.

    Args:
        file_path: Path to the gzipped file.

    Returns:
        None
    """
    with gzip.open(file_path, "rb") as f_in:
        with open(file_path[:-3], "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def _read_file(file_path: str):
    """
    Reads a file based on the file extension.

    Args:
        file_path: Path to the file.

    Returns:
        lines: List of lines in the file.
    """
    log.debug(f"Reading file {file_path}")
    file_extension = file_path.split(".")[-1]

    # Read file based on extension
    match file_extension:
        case "txt":
            return _read_txt(file_path)
        case "json":
            # Special case
            if file_path.endswith("class_vector.json"):
                return _read_class_vector(file_path)
            return _read_json(file_path)
        case "csv":
            return _read_csv(file_path)
        case "pt":
            return _read_torch(file_path)
        case _:
            raise ValueError(f"Unknown file extension: {file_extension}")


def _read_txt(file_path: str) -> list[str]:
    """
    Reads a text file and returns the lines as a list of strings.

    Args:
        file_path: Path to the text file.

    Returns:
        lines: List of lines in the text file.
    """
    with open(file_path, "r") as f:
        return list(map(lambda x: x.strip(), f.readlines()))


def _read_csv(file_path: str) -> pd.DataFrame:
    """
    Reads a csv file and returns a pandas DataFrame.

    Args:
        file_path: Path to the csv file.

    Returns:
        df: Pandas DataFrame containing the csv data.
    """
    return pd.read_csv(file_path)


def _read_json(file_path: str) -> dict:
    """
    Reads a json file and returns a dictionary.

    Args:
        file_path: Path to the json file.

    Returns:
        dict: Dictionary containing the json data.
    """
    with open(file_path, "r") as f:
        return json.load(f)


def _read_torch(file_path: str) -> torch.nn.Module:
    """
    Reads a model file and returns a torch.nn.Module to CPU.

    Args:
        file_path: Path to the model file.

    Returns:
        model: torch.nn.Module
    """
    return torch.load(file_path, map_location=torch.device("cpu"))


def _read_class_vector(file_path: str) -> dict:
    """
    Reads a the class_vector.json file because it's
    not valid JSON

    Args:
        file_path: Path to the json file.

    Returns:
        dict: Dictionary containing the json data.
    """
    with open(file_path, "r") as f:
        json_objects = f.read().split("\n")

    parsed_objects = []
    for obj in json_objects:
        try:
            parsed_objects.append(json.loads(obj))
        except json.JSONDecodeError:
            print("Invalid JSON object:", obj)
            # Handle the error or continue
    return parsed_objects
