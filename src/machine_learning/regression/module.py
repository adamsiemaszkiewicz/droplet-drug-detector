# -*- coding: utf-8 -*-
from typing import Dict, Optional, Tuple, Union

from lightning import LightningModule
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import MeanSquaredError, MetricCollection, R2Score

from src.common.consts.machine_learning import STAGE_TESTING, STAGE_TRAINING, STAGE_VALIDATION
from src.machine_learning.augmentations.config import AugmentationsConfig
from src.machine_learning.augmentations.factory import create_augmentations
from src.machine_learning.optimizer.config import OptimizerConfig
from src.machine_learning.optimizer.factory import create_optimizer
from src.machine_learning.regression.loss_functions.config import RegressionLossFunctionConfig
from src.machine_learning.regression.loss_functions.factory import create_loss_function
from src.machine_learning.regression.models.config import RegressionModelConfig
from src.machine_learning.regression.models.factory import create_model
from src.machine_learning.scheduler.config import SchedulerConfig
from src.machine_learning.scheduler.factory import create_scheduler


class RegressionLightningModule(LightningModule):
    """
    PyTorch Lightning module for a regression task.

    This class extends LightningModule and is configured via a RegressionConfig object.
    It defines the forward pass, training step, validation step, and test step, as well as
    the optimizer and learning rate scheduler setup.

    Args:
        model_config (ClassificationModelConfig): The model configuration.
        loss_function_config (ClassificationLossFunctionConfig): The loss function configuration.
        optimizer_config (OptimizerConfig): The optimizer configuration.
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
        model_config: RegressionModelConfig,
        loss_function_config: RegressionLossFunctionConfig,
        optimizer_config: OptimizerConfig,
        augmentations_config: Optional[AugmentationsConfig] = None,
        scheduler_config: Optional[SchedulerConfig] = None,
    ):
        super().__init__()
        self.model_config = model_config
        self.loss_function_config = loss_function_config
        self.optimizer_config = optimizer_config
        self.augmentations_config = augmentations_config
        self.scheduler_config = scheduler_config

        self.model = create_model(config=model_config)
        self.loss_function = create_loss_function(config=loss_function_config)
        self.metrics = MetricCollection(
            [
                MeanSquaredError(),
                R2Score(),
            ]
        )
        self.train_metrics = self.metrics.clone(prefix=f"{STAGE_TRAINING}_")
        self.val_metrics = self.metrics.clone(prefix=f"{STAGE_VALIDATION}_")
        self.test_metrics = self.metrics.clone(prefix=f"{STAGE_TESTING}_")

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

        preds = self(x).squeeze()
        targets = y.float()
        per_sample_losses = self.compute_loss(preds=preds, targets=targets)
        loss = per_sample_losses.mean()

        self.log(name=f"{stage}_loss", value=loss)

        if stage == STAGE_TRAINING:
            metrics = self.train_metrics
        elif stage == STAGE_VALIDATION:
            metrics = self.val_metrics
        else:
            metrics = self.test_metrics

        output = metrics(preds, y)
        self.log_dict(output, on_step=(stage == STAGE_TRAINING), on_epoch=True)

        return {"loss": loss, "per_sample_losses": per_sample_losses, "preds": preds, "targets": targets}

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
        self.compute_and_log_metrics(self.train_metrics)

    def on_validation_epoch_end(self) -> None:
        self.compute_and_log_metrics(self.val_metrics)

    def on_test_epoch_end(self) -> None:
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
        Compute the loss based on logits and targets.

        Args:
            preds (Tensor): The predicted values.
            targets (Tensor): The true values.

        Returns:
            Tensor: The computed loss.
        """
        return self.loss_function(preds, targets)
