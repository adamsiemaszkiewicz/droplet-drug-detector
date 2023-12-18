# -*- coding: utf-8 -*-
from typing import Dict, Optional, Tuple, Union

from lightning import LightningModule
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.common.consts.machine_learning import STAGE_TESTING, STAGE_TRAINING, STAGE_VALIDATION
from src.machine_learning.augmentations.config import AugmentationsConfig
from src.machine_learning.augmentations.factory import create_augmentations
from src.machine_learning.classification.loss_functions.config import ClassificationLossFunctionConfig
from src.machine_learning.classification.loss_functions.factory import create_loss_function
from src.machine_learning.classification.metrics.config import ClassificationMetricsConfig
from src.machine_learning.classification.metrics.factory import create_metrics
from src.machine_learning.classification.models.config import ClassificationModelConfig
from src.machine_learning.classification.models.factory import create_model
from src.machine_learning.optimizer.config import OptimizerConfig
from src.machine_learning.optimizer.factory import create_optimizer
from src.machine_learning.scheduler.config import SchedulerConfig
from src.machine_learning.scheduler.factory import create_scheduler


class ClassificationLightningModule(LightningModule):
    """
    PyTorch Lightning module for a classification task.

    This class extends LightningModule and is configured via a ClassificationConfig object.
    It defines the forward pass, training step, validation step, and test step, as well as
    the optimizer and learning rate scheduler setup.

    Args:
        classes (Dict[int, str]): A dictionary mapping class indices to class names.
        model_config (ClassificationModelConfig): The model configuration.
        loss_function_config (ClassificationLossFunctionConfig): The loss function configuration.
        optimizer_config (OptimizerConfig): The optimizer configuration.
        metrics_config (ClassificationMetricsConfig): The metrics configuration.
        augmentations_config (Optional[AugmentationsConfig]): The data augmentations configuration.
        scheduler_config (Optional[SchedulerConfig]): The scheduler configuration.

    Attrs:
        model_config (ClassificationModelConfig): The model configuration.
        loss_function_config (ClassificationLossFunctionConfig): The loss function configuration.
        optimizer_config (OptimizerConfig): The optimizer configuration.
        metrics_config (ClassificationMetricsConfig): The metrics configuration.
        augmentations_config (Optional[AugmentationsConfig]): The data augmentations configuration.
        scheduler_config (Optional[SchedulerConfig]): The scheduler configuration.
        model (Module): The neural network model defined by the create_model function.
        loss_function (Module): The loss function, instantiated based on config.
        metrics (Dict[str, Metric]): A dictionary mapping metric names to Metric instances.
        augmentations (Optional[Module]): An optional data augmentation module.
    """

    def __init__(
        self,
        classes: Dict[int, str],
        model_config: ClassificationModelConfig,
        loss_function_config: ClassificationLossFunctionConfig,
        optimizer_config: OptimizerConfig,
        metrics_config: ClassificationMetricsConfig,
        augmentations_config: Optional[AugmentationsConfig] = None,
        scheduler_config: Optional[SchedulerConfig] = None,
    ):
        super().__init__()
        self.classes = classes
        self.model_config = model_config
        self.loss_function_config = loss_function_config
        self.optimizer_config = optimizer_config
        self.metrics_config = metrics_config
        self.augmentations_config = augmentations_config
        self.scheduler_config = scheduler_config

        self.model = create_model(config=model_config)
        self.loss_function = create_loss_function(config=loss_function_config)
        self.metrics = create_metrics(config=metrics_config)
        self.augmentations = create_augmentations(config=augmentations_config) if augmentations_config else None

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
        per_sample_losses = self.compute_loss(logits=logits, targets=y)
        loss = per_sample_losses.mean()
        preds = logits.argmax(dim=1)

        self.log(name=f"{stage}_loss", value=loss)
        self.log_metrics(preds=preds, targets=y, stage=stage)

        return {"loss": loss, "per_sample_losses": per_sample_losses, "preds": preds, "targets": y}

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

    def on_after_batch_transfer(self, batch: Tuple[Tensor, Tensor], dataloader_idx: int) -> Tuple[Tensor, Tensor]:
        """
        Hook to perform operations on the batch after the batch transfer but before passing to the model.

        If the model is in training mode and an augmenter is provided, it will apply augmentation
        to the batch of data.

        Args:
            batch (Tuple[Tensor, Tensor]): A tuple containing a batch of input data and labels.
            dataloader_idx (int): The index of the dataloader that provided the batch.

        Returns:
            Tuple[Tensor, Tensor]: A tuple of augmented data and labels, or the original batch
            if no augmentation is to be applied.
        """
        x, y = batch
        if self.trainer.training and self.augmentations:
            x = self.augmentations(x)
        return x, y

    def configure_optimizers(self) -> Dict[str, Union[Optimizer, LRScheduler]]:
        """
        Configure the optimizer using the model parameters and specified learning rate.

        Returns:
            Dict[str, Union[Optimizer, LRScheduler]]: Optimizer and optional learning rate scheduler
        """
        optimizer = create_optimizer(config=self.optimizer_config, parameters=self.parameters())
        optimizers_config = {"optimizer": optimizer}

        if self.scheduler_config:
            total_steps = int(self.trainer.estimated_stepping_batches)

            scheduler = create_scheduler(config=self.scheduler_config, optimizer=optimizer, total_steps=total_steps)
            optimizers_config["lr_scheduler"] = scheduler

        return optimizers_config

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

    def log_metrics(self, preds: Tensor, targets: Tensor, stage: str) -> None:
        """
        Log metrics to the logger, including class-wise metrics.

        Args:
            preds (Tensor): The model predictions after applying argmax on logits.
            targets (Tensor): The true labels.
            stage (str): The current stage (e.g., training, validation, or testing).
        """
        for name, metric in self.metrics.items():
            # Overall metric
            metric_value = metric(preds, targets)
            self.log(name=f"{stage}_{name}", value=metric_value, on_step=True, on_epoch=True)

            # Class-wise metrics
            for class_idx, class_name in self.classes.items():
                class_targets = (targets == class_idx).int()
                class_preds = (preds == class_idx).int()
                class_metric_value = metric(class_preds, class_targets)

                self.log(
                    name=f"{stage}_{name}_class_{class_idx}_{class_name}",
                    value=class_metric_value,
                    on_step=True,
                    on_epoch=True,
                )
