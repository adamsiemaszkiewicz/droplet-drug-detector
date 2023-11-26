# -*- coding: utf-8 -*-
from lightning import seed_everything
from lightning.pytorch.loggers import WandbLogger

from src.aml.components.classificator_training.arg_parser import get_config
from src.aml.components.classificator_training.config import ClassificationConfig
from src.aml.components.classificator_training.data import ClassificationDataModule
from src.common.utils.logger import get_logger
from src.machine_learning.callbacks.factory import create_callbacks
from src.machine_learning.classification.module import ClassificationLightningModule
from src.machine_learning.loggers.factory import create_loggers
from src.machine_learning.preprocessing.factory import create_preprocessor
from src.machine_learning.trainer.factory import create_trainer

_logger = get_logger(__name__)


def main() -> None:
    config: ClassificationConfig = get_config()
    _logger.info(f"Running with following config: {config}")

    seed_everything(seed=config.seed, workers=True)

    preprocessor = create_preprocessor(config=config.preprocessing) if config.preprocessing else None
    dm = ClassificationDataModule(config=config.data, preprocessor=preprocessor)

    model = ClassificationLightningModule(
        model_config=config.model,
        loss_function_config=config.loss_function,
        optimizer_config=config.optimizer,
        metrics_config=config.metrics,
        augmentations_config=config.augmentations,
        scheduler_config=config.scheduler,
    )

    callbacks = create_callbacks(config=config.callbacks) if config.callbacks else None
    loggers = create_loggers(config=config.loggers) if config.loggers else None

    if loggers is not None:
        for logger in loggers:
            if isinstance(logger, WandbLogger):
                logger.experiment.config.update(config.dict())
                break

    trainer = create_trainer(config=config.trainer, callbacks=callbacks, loggers=loggers)
    trainer.fit(model=model, datamodule=dm)

    best_model_path = trainer.checkpoint_callback.best_model_path

    if not best_model_path:
        raise FileNotFoundError("No best model checkpoint found.")

    _logger.info(f"Loading best model for testing purposes from: {best_model_path}")
    best_model = ClassificationLightningModule.load_from_checkpoint(
        checkpoint_path=best_model_path,
        model_config=config.model,
        loss_function_config=config.loss_function,
        optimizer_config=config.optimizer,
        metrics_config=config.metrics,
        augmentations_config=config.augmentations,
        scheduler_config=config.scheduler,
    )
    trainer.test(model=best_model, datamodule=dm)


if __name__ == "__main__":
    main()
