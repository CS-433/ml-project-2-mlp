"""
Script for evaluating the model.
"""

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="eval")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
