"""
Script for training a model on a dataset.
"""

from typing import List

import hydra
import lightning as L
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

import ml_project_2_mlp.utils as utils
import wandb
from ml_project_2_mlp.data import WebsiteData
from ml_project_2_mlp.labeler import WebsiteLabeler
from ml_project_2_mlp.logger import RankedLogger

# Setup root environment
root_path = rootutils.setup_root(__file__)
rootutils.set_root(
    path=root_path,
    project_root_env_var=True,
)


# Setup ranked logger
log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: DictConfig):
    # Extras (e.g. ignore warnings, pretty print config, ...)
    utils.extras(cfg)

    # Set all seeds
    if cfg.get("seed"):
        L.seed_everything(cfg.seed)

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

    # Instantiate model module
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Instantiating callbacks for Lightning Trainer
    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    # Instantiating loggers for Lightning Trainer
    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    # Instantiate trainer module
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    setup_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    # Log hyperparameters if specified
    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(setup_dict)

    # Train model
    if cfg.finetune:
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)

        train_metrics = trainer.callback_metrics
    else:
        train_metrics = {}
        log.info("Skipping finetuning!")

    # Test model if specified
    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    metric_dict = {**train_metrics, **test_metrics}

    # For hydra-optuna integration
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # Finish logging
    wandb.finish()

    return metric_value


if __name__ == "__main__":
    main()
