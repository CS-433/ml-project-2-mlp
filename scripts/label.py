"""
Script to label an existing dataset using a specified
labeler. Configurations for this script are in `conf/label.yaml`.
"""

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
    print(cfg)

    # Instantiate data
    data = hydra.utils.instantiate(cfg.data)
    print(data)

    # Instantiate labeler
    labeler = hydra.utils.instantiate(cfg.labeler, data=data)
    print(labeler)


if __name__ == "__main__":
    main()
