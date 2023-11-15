# -*- coding: utf-8 -*-
from lightning import seed_everything

from src.common.consts.directories import CONFIGS_DIR
from src.common.utils.logger import get_logger
from src.configs.classification import ClassificationMachineLearningConfig
from src.machine_learning.callbacks import create_callbacks
from src.machine_learning.classification.lightning_module import ClassificationLightningModule
from src.machine_learning.data import ClassificationDataModule
from src.machine_learning.loggers import create_loggers
from src.machine_learning.preprocessing import DataPreprocessor
from src.machine_learning.trainer import create_trainer

_logger = get_logger(__name__)


def main() -> None:
    config_file_path = CONFIGS_DIR / "base.yaml"
    config = ClassificationMachineLearningConfig.from_yaml(path=config_file_path)

    seed_everything(42, workers=True)

    preprocessor = DataPreprocessor(config=config.preprocessing) if config.preprocessing else None
    dm = ClassificationDataModule(config=config.data, preprocessor=preprocessor)

    model = ClassificationLightningModule(config=config)

    callbacks = create_callbacks(config=config.callbacks) if config.callbacks else None
    loggers = create_loggers(config=config.loggers) if config.loggers else None

    trainer = create_trainer(config=config.trainer, callbacks=callbacks, loggers=loggers)
    trainer.fit(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
