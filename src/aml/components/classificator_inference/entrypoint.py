# -*- coding: utf-8 -*-
from pathlib import Path

import torch
from lightning import seed_everything

from src.aml.components.classificator_training.arg_parser import get_config
from src.aml.components.classificator_training.config import ClassificationConfig
from src.aml.components.classificator_training.data.dataset import DropletDrugClassificationDataset
from src.common.utils.logger import get_logger
from src.machine_learning.classification.module import ClassificationLightningModule
from src.machine_learning.preprocessing.factory import create_preprocessor

_logger = get_logger(__name__)


def main() -> None:
    config: ClassificationConfig = get_config()
    _logger.info(f"Running with following config: {config}")

    seed_everything(seed=config.seed, workers=True)

    preprocessor = create_preprocessor(config=config.preprocessing) if config.preprocessing else None

    best_model_path = (
        "/home/adam/PycharmProjects/droplet-drug-detector/artifacts/droplet-drug-classificator/"
        "2023-11-28_17-52-43/checkpoints/epoch=4-val_loss=0.0416.ckpt"
    )

    if not best_model_path:
        raise FileNotFoundError("No best model checkpoint found.")

    _logger.info(f"Loading best model for testing purposes from: {best_model_path}")
    model = ClassificationLightningModule.load_from_checkpoint(
        checkpoint_path=best_model_path,
        classes=DropletDrugClassificationDataset.CLASSES,
        model_config=config.model,
        loss_function_config=config.loss_function,
        optimizer_config=config.optimizer,
        metrics_config=config.metrics,
        augmentations_config=config.augmentations,
        scheduler_config=config.scheduler,
    )

    model.eval()
    model.to("cpu")

    image_paths = [
        Path("/home/adam/PycharmProjects/droplet-drug-detector/data/dataset/lactose_0.25mgml_x40/2.jpg"),
        Path("/home/adam/PycharmProjects/droplet-drug-detector/data/dataset/naproxen_0.25mgml_x40/22.jpg"),
        Path("/home/adam/PycharmProjects/droplet-drug-detector/data/dataset/pearlitol_0.25mgml_x40/17.jpg"),
        Path("/home/adam/PycharmProjects/droplet-drug-detector/data/dataset/pearlitol_0.5mgml_x40/21.jpg"),
        Path("/home/adam/PycharmProjects/droplet-drug-detector/data/dataset/naproxen_0.25mgml_x40/12.jpg"),
        Path("/home/adam/PycharmProjects/droplet-drug-detector/data/dataset/lactose_0.25mgml_x40/13.jpg"),
    ]
    labels = [1, 3, 4, 4, 3, 1]

    prediction_dataset = DropletDrugClassificationDataset(
        image_paths=image_paths, labels=labels, transform=preprocessor
    )

    with torch.no_grad():
        for input, _ in prediction_dataset:
            logits = model(input.unsqueeze(0))
            proba = torch.softmax(logits, dim=1)
            class_id = torch.argmax(proba, dim=1).item()
            _logger.info(f"Prediction: {class_id}")


if __name__ == "__main__":
    main()
