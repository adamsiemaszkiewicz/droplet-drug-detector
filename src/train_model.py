# -*- coding: utf-8 -*-
from lightning import seed_everything

from src.common.consts.directories import CONFIGS_DIR
from src.common.utils.logger import get_logger
from src.configs.classification import ClassificationConfig
from src.machine_learning.callbacks.factory import create_callbacks
from src.machine_learning.data import ClassificationDataModule
from src.machine_learning.lightning_module.classification.module import ClassificationLightningModule
from src.machine_learning.loggers.factory import create_loggers
from src.machine_learning.preprocessing.factory import create_preprocessor
from src.machine_learning.trainer.factory import create_trainer

_logger = get_logger(__name__)


def main() -> None:
    config_file_path = CONFIGS_DIR / "base.yaml"
    config = ClassificationConfig.from_yaml(path=config_file_path)

    seed_everything(seed=42, workers=True)

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

    trainer = create_trainer(config=config.trainer, callbacks=callbacks, loggers=loggers)
    trainer.fit(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
