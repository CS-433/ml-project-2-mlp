"""
Script to label an existing dataset using a specified
labeler. Configurations for this script are in `conf/label.yaml`.
"""

import hydra
import rootutils
from omegaconf import DictConfig

import ml_project_2_mlp.utils as utils
from ml_project_2_mlp.data import WebsiteData
from ml_project_2_mlp.logger import RankedLogger

# Setup root environment
root_path = rootutils.setup_root(__file__)
rootutils.set_root(
    path=root_path,
    project_root_env_var=True,
)

# Setup ranked logger
log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base=None, config_path="../conf", config_name="scrape")
def main(cfg: DictConfig):
    # Initialize the WebsiteData
    hydra.utils.instantiate(cfg.data)


if __name__ == "__main__":
    main()
