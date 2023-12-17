"""
Script for evaluating the model.
"""
from typing import List

import hydra
import lightning as L
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

import ml_project_2_mlp.utils as utils
import wandb
from ml_project_2_mlp.data import WebsiteData
from ml_project_2_mlp.labeler import WebsiteLabeler
from ml_project_2_mlp.logger import RankedLogger
from ml_project_2_mlp.utils import instantiate_loggers

# Setup root environment
root_path = rootutils.setup_root(__file__)
rootutils.set_root(
    path=root_path,
    project_root_env_var=True,
)

# Setup rankedd logger
log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base=None, config_path="../conf", config_name="eval")
def main(cfg: DictConfig):
    """
    Evaluates a trained model (from a PyTorch Lightning checkpoint that is saved
    as an artifact on Weights & Biases on the test dataloader)
    dataset.

    To run, specify the run ID of the experiment and the model version to evaluate
    on (can be either `latest`, `best`, or a specific version number). For example,
    to evaluate the best model from the experiment with run ID `1`, run the following
    command:

        poetry run eval run_id=1 version=best
    """
    # Print experiment configuration
    utils.extras(cfg)

    # Set all seeds
    if cfg.get("seed"):
        L.seed_everything(cfg.seed)

    # Download the checkpoint locally
    run = wandb.init(
        project=cfg.logger.wandb.project,
        entity=cfg.logger.wandb.entity,
        id=cfg.run_id,
        resume="must",
        job_type="eval",
    )

    # Instantiate data
    log.info(f"Instantiating data <{cfg.data._target_}>")
    data: WebsiteData = hydra.utils.instantiate(cfg.data)

    # Instantiate labeler
    log.info(f"Instantiating labeler <{cfg.labeler._target_}>")
    labeler: WebsiteLabeler = hydra.utils.instantiate(cfg.labeler, data=data)

    # Instantiate data module
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.datamodule, data=data, labeler=labeler
    )

    # Instantaite model module
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Instantiating loggers for Lightning Trainer
    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # Instantiate trainer module
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    log.info("Testing model!")
    ckpt_dir = run.config.best_model_path
    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_dir)

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
