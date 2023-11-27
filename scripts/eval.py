"""
Script for evaluating the model.
"""
import os
from typing import List

import hydra
import lightning as L
import rootutils
import wandb
from lightning import LightningDataModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

import ml_project_2_mlp.utils as utils
from ml_project_2_mlp.logger import RankedLogger
from ml_project_2_mlp.model import Homepage2VecModule
from ml_project_2_mlp.utils import instantiate_loggers, log_hyperparameters

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
    as an artifact on Weights & Biases on the test set split of the crowdsourced
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
        job_type="eval",
    )
    artifact = run.use_artifact(cfg.wandb_ckpt, type="model")
    artifact_dir = artifact.download()

    # Load homepage2vec module from checkpoint
    model = Homepage2VecModule.load_from_checkpoint(
        os.path.join(artifact_dir, "model.ckpt")
    )

    # Instantiate data module
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Instantiating loggers for Lightning Trainer
    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # Instantiate trainer module
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    setup_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(setup_dict)

    log.info("Testing model!")
    trainer.test(model=model, datamodule=datamodule)

    # log.info("Predicting on test split!")
    # preds = trainer.predict(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
