"""
Script for training a model on a dataset.
"""

from typing import List

import hydra
import lightning as L
import rootutils
import torch
import wandb
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

import ml_project_2_mlp.utils as utils
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

    # Instantiate train data
    log.info(f"Instantiating train data <{cfg.train_data._target_}>")
    train_data: WebsiteData = hydra.utils.instantiate(cfg.train_data)

    # Instantiate labeler
    log.info(f"Instantiating train labeler <{cfg.train_labeler._target_}>")
    train_labeler: WebsiteLabeler = hydra.utils.instantiate(
        cfg.train_labeler, data=train_data
    )

    # Instantiate train data module
    log.info(f"Instantiating training datamodule <{cfg.train_datamodule._target_}>")
    train_datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.train_datamodule, data=train_data, labeler=train_labeler
    )
    train_datamodule.setup()

    log.info(f"Instantiating test data <{cfg.test_data._target_}>")
    test_data: WebsiteData = hydra.utils.instantiate(cfg.test_data)

    # Instantiate labeler
    log.info(f"Instantiating test labeler <{cfg.test_labeler._target_}>")
    test_labeler: WebsiteLabeler = hydra.utils.instantiate(
        cfg.test_labeler, data=test_data
    )

    log.info(f"Instantiating testing datamodule <{cfg.test_datamodule._target_}>")
    test_datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.test_datamodule, data=test_data, labeler=test_labeler
    )
    test_datamodule.setup()

    # Instantiate model module
    log.info(f"Instantiating model <{cfg.model._target_}>")
    train_data = train_datamodule.train_dataset.dataset
    train_labels = train_data.labels
    num_samples = len(train_labels)
    class_counts = train_labels.sum(axis=0)
    # num_samples = 886000
    # train_ratio = torch.Tensor(
    #     [
    #         0.093,
    #         0.276,
    #         0.062,
    #         0.017,
    #         0.059,
    #         0.015,
    #         0.011,
    #         0.011,
    #         0.084,
    #         0.043,
    #         0.048,
    #         0.074,
    #         0.139,
    #         0.068,
    #     ]
    # )
    # class_counts = train_ratio * num_samples
    pos_ratio = (num_samples - class_counts) / class_counts

    model: LightningModule = hydra.utils.instantiate(cfg.model, pos_ratio=pos_ratio)

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
        "model": model,
        "trainer": trainer,
    }

    # Log hyperparameters if specified
    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(setup_dict)

    # Train model
    if cfg.get("finetune"):
        log.info("Starting training!")
        train_loader = train_datamodule.train_dataloader()
        val_loader = test_datamodule.val_dataloader()
        trainer.fit(
            model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )

        train_metrics = trainer.callback_metrics
    else:
        train_metrics = {}
        log.info("Skipping finetuning!")

    # Logging best model path
    ckpt_path = trainer.checkpoint_callback.best_model_path
    log.info("Saving best model path to hyperparameters!")
    for logger in trainer.loggers:
        logger.log_hyperparams({"best_model_path": ckpt_path})

    # Test model if specified
    if cfg.get("eval"):
        log.info("Starting testing!")
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        test_loader = test_datamodule.test_dataloader()
        trainer.test(
            model=model,
            dataloaders=test_loader,
            ckpt_path=ckpt_path,
        )
        test_metrics = trainer.callback_metrics
        log.info(f"Best ckpt path: {ckpt_path}")
    else:
        test_metrics = {}
        log.info("Skipping testing!")

    # For hydra-optuna integration
    metric_dict = {**train_metrics, **test_metrics}
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # Finish logging
    wandb.finish()

    return metric_value


if __name__ == "__main__":
    main()
