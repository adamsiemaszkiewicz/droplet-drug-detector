# -*- coding: utf-8 -*-
from typing import Dict, Tuple

from lightning import LightningModule
from torch import Tensor
from torch.optim import Optimizer

from src.common.consts.machine_learning import STAGE_TESTING, STAGE_TRAINING, STAGE_VALIDATION
from src.common.machine_learning.augmentations import create_augmentations
from src.common.machine_learning.classification.config import ClassificationConfig
from src.common.machine_learning.classification.loss_functions import create_loss_function
from src.common.machine_learning.classification.metrics import create_metrics
from src.common.machine_learning.classification.models import create_model
from src.common.machine_learning.optimizer import create_optimizer
from src.common.machine_learning.scheduler import create_scheduler


class ClassificationLightningModule(LightningModule):
    def __init__(self, config: ClassificationConfig):
        super().__init__()

        self.model = create_model(config=config.model)
        self.loss_function = create_loss_function(config=config.loss_function)
        self.optimizer = create_optimizer(config=config.optimizer)
        self.scheduler = create_scheduler(config=config.scheduler, optimizer=self.optimizer)
        self.metrics = create_metrics(config=config.metrics)
        self.augmentations = create_augmentations(config=config.augmentations)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def evaluation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int, stage: str) -> Dict[str, Tensor]:
        """
        Shared evaluation step for validation and testing.

        Args:
            batch (Tuple[Tensor, Tensor]): The current batch of inputs, labels & reference GeoTIFFs.
            batch_idx (int): Index of the current batch.
            stage (str): Current stage (e.g., validation or testing).

        Returns:
            Dict[str, Tensor]: Loss & predictions for the current batch.
        """
        x, y = batch

        logits = self(x)
        loss = self.compute_loss(logits, y)
        self.log(f"{stage}_loss", loss)

        self.log_metrics(logits=logits, targets=y, stage=stage)

        preds = logits.argmax(dim=1)

        return {"loss": loss, "preds": preds}

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        """
        Defines the procedure of a single training step.

        Args:
            batch (Tuple[Tensor, Tensor): The current batch of training data, labels & reference GeoTIFFs.
            batch_idx (int): Index of the current batch.

        Returns:
            Dict[str, Tensor]: Training loss.
        """
        return self.evaluation_step(batch=batch, batch_idx=batch_idx, stage=STAGE_TRAINING)

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        """
        Defines the procedure of a single validation step.

        Args:
            batch (Tuple[Tensor, Tensor]): The current batch of validation data and labels.
            batch_idx (int): Index of the current batch.

        Returns:
            Dict[str, Tensor]: Validation loss.
        """
        return self.evaluation_step(batch=batch, batch_idx=batch_idx, stage=STAGE_VALIDATION)

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        """
        Defines the procedure of a single testing step.

        Args:
            batch (Tuple[Tensor, Tensor]): The current batch of testing data and labels.
            batch_idx (int): Index of the current batch.

        Returns:
            Dict[str, Tensor]: Testing loss.
        """
        return self.evaluation_step(batch=batch, batch_idx=batch_idx, stage=STAGE_TESTING)

    def on_after_batch_transfer(self, batch: Tensor, dataloader_idx: int) -> Tuple[Tensor, Tensor]:
        x, y = batch
        if self.trainer.training:
            x = self.augmentations(x)
        return x, y

    def configure_optimizers(self) -> Dict[str, Optimizer]:
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Compute the loss based on logits and targets.

        Args:
            logits (Tensor): The model outputs before activation.
            targets (Tensor): The true labels.

        Returns:
            Tensor: The computed loss.
        """
        return self.loss_function(logits, targets)

    def log_metrics(self, logits: Tensor, targets: Tensor, stage: str) -> None:
        """
        Log metrics to the logger.

        Args:
            logits (Tensor): The model outputs before activation.
            targets (Tensor): The true labels.
            stage (str): The current stage (e.g., training, validation, or testing).
        """
        for name, metric in self.metrics.items():
            metric_value = metric(logits, targets)
            self.log(f"{stage}_{name}", metric_value, on_step=True, on_epoch=True)
