"""
Script for training a model on a dataset.
"""

import logging
from typing import List

import hydra
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

import ml_project_2_mlp.utils as utils

# Setup root environment
root_path = rootutils.setup_root(__file__)
rootutils.set_root(
    path=root_path,
    project_root_env_var=True,
)

log = logging.Logger(__name__, level=logging.INFO)


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: DictConfig):
    # Print experiment configuration
    print(OmegaConf.to_yaml(cfg))

    # Set all seeds
    if cfg.get("seed"):
        utils.seed_everything(cfg.seed)

    # Instantiate data module
    # log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    # datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

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

    return

    setup_dict = {
        "cfg": cfg,
        # "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    # Train model if specified
    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)

    train_metrics = trainer.callback_metrics

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

    # Merge train and test metrics
    result_dict = {**train_metrics, **test_metrics}

    return setup_dict, result_dict


if __name__ == "__main__":
    main()
