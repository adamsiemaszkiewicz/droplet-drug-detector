# -*- coding: utf-8 -*-
from typing import Any, Dict, Optional, Tuple, Union

from lightning import LightningModule
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import MetricCollection

from src.common.consts.machine_learning import STAGE_TESTING, STAGE_TRAINING, STAGE_VALIDATION
from src.machine_learning.augmentations.config import AugmentationsConfig
from src.machine_learning.augmentations.factory import create_augmentations
from src.machine_learning.loss_functions.config import BaseLossFunctionConfig
from src.machine_learning.loss_functions.factory import create_loss_function
from src.machine_learning.models.config import BaseModelConfig
from src.machine_learning.models.factory import create_model
from src.machine_learning.optimizer.config import OptimizerConfig
from src.machine_learning.optimizer.factory import create_optimizer
from src.machine_learning.scheduler.config import SchedulerConfig
from src.machine_learning.scheduler.factory import create_scheduler


class BaseLightningModule(LightningModule):
    def __init__(
        self,
        model_config: BaseModelConfig,
        loss_function_config: BaseLossFunctionConfig,
        optimizer_config: OptimizerConfig,
        augmentations_config: Optional[AugmentationsConfig] = None,
        scheduler_config: Optional[SchedulerConfig] = None,
    ):
        """
        This is a base class for Pytorch-Lightning-based deep learning tasks.

        Args:
            model_config (BaseModelConfig): Configuration for the model.
            loss_function_config (BaseLossFunctionConfig): Configuration for the loss function.
            optimizer_config (OptimizerConfig): Configuration for the optimizer.
            augmentations_config (Optional[AugmentationsConfig]): Optional configuration for data augmentations.
            scheduler_config (Optional[SchedulerConfig]): Optional configuration for learning rate scheduler.
        """
        super().__init__()
        self.model = create_model(model_config)

        self.loss_function = create_loss_function(loss_function_config)

        self.metrics = self.setup_metrics()
        self.train_metrics = self.metrics.clone(prefix=f"{STAGE_TRAINING}_")
        self.val_metrics = self.metrics.clone(prefix=f"{STAGE_VALIDATION}_")
        self.test_metrics = self.metrics.clone(prefix=f"{STAGE_TESTING}_")

        self.augmentations = create_augmentations(augmentations_config) if augmentations_config else None

        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor from the model.
        """
        return self.model(x)

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

    def compute_and_log_metrics(self, metrics: MetricCollection) -> None:
        """
        Compute and log metrics for a given stage, and reset them.

        Args:
            metrics (MetricCollection): The metrics to compute and log.
        """
        output = metrics.compute()
        self.log_dict(output)
        metrics.reset()

    def on_train_epoch_end(self) -> None:
        """
        Hook to perform operations at the end of the training epoch.
        """
        self.compute_and_log_metrics(self.train_metrics)

    def on_validation_epoch_end(self) -> None:
        """
        Hook to perform operations at the end of the validation epoch.
        """
        self.compute_and_log_metrics(self.val_metrics)

    def on_test_epoch_end(self) -> None:
        """
        Hook to perform operations at the end of the testing epoch.
        """
        self.compute_and_log_metrics(self.test_metrics)

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

    def compute_loss(self, preds: Tensor, targets: Tensor) -> Tensor:
        """
        Compute the loss based on predictions and targets.

        Args:
            preds (Tensor): The model predictions.
            targets (Tensor): The true labels.

        Returns:
            Tensor: The computed loss.
        """
        return self.loss_function(preds, targets)

    # Abstract methods that need to be implemented by derived classes
    def evaluation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int, stage: str) -> Dict[str, Any]:
        """
        Abstract method for performing an evaluation step. This should be implemented in derived classes.

        Args:
            batch (Tuple[Tensor, Tensor]): The current batch of data and labels.
            batch_idx (int): Index of the current batch.
            stage (str): Stage of evaluation (training, validation, or testing).

        Returns:
            Dict[str, Any]: A dictionary containing evaluation results like loss.
        """
        raise NotImplementedError

    def setup_metrics(self) -> MetricCollection:
        """
        Abstract method for setting up metrics. This should be implemented in derived classes to define metrics.

        Returns:
            MetricCollection: A collection of metrics.
        """
        raise NotImplementedError
