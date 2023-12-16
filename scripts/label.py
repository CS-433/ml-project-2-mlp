"""
Script to label an existing dataset using a specified
labeler. Configurations for this script are in `conf/label.yaml`.
"""

from cProfile import label

import hydra
import rootutils
from omegaconf import DictConfig, OmegaConf

from ml_project_2_mlp.logger import RankedLogger

# Setup root environment
root_path = rootutils.setup_root(__file__)
rootutils.set_root(
    path=root_path,
    project_root_env_var=True,
)

# Setup ranked logger
log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base=None, config_path="../conf", config_name="label")
def main(cfg: DictConfig):
    cfg = OmegaConf.create(cfg)

    # Instantiate data and labeler
    print(f"Initialising dataset: {cfg.data.name}")
    data = hydra.utils.instantiate(cfg.data)

    print(f"Initialising labeler: {cfg.labeler.name}")
    labeler = hydra.utils.instantiate(cfg.labeler, data=data)

    # Get the labels
    num_labels = len(labeler.get_labels())
    print(f"Done! Labeled {num_labels} websites.")


if __name__ == "__main__":
    main()
