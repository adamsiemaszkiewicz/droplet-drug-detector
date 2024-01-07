# -*- coding: utf-8 -*-
from typing import Dict, Optional, Tuple

from torch import Tensor
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MeanSquaredLogError, MetricCollection, R2Score

from src.common.consts.machine_learning import STAGE_TRAINING, STAGE_VALIDATION
from src.machine_learning.augmentations.config import AugmentationsConfig
from src.machine_learning.loss_functions.config import RegressionLossFunctionConfig
from src.machine_learning.models.config import RegressionModelConfig
from src.machine_learning.modules.base import BaseLightningModule
from src.machine_learning.optimizer.config import OptimizerConfig
from src.machine_learning.scheduler.config import SchedulerConfig


class RegressionLightningModule(BaseLightningModule):
    def __init__(
        self,
        model_config: RegressionModelConfig,
        loss_function_config: RegressionLossFunctionConfig,
        optimizer_config: OptimizerConfig,
        augmentations_config: Optional[AugmentationsConfig] = None,
        scheduler_config: Optional[SchedulerConfig] = None,
    ):
        super().__init__(
            model_config=model_config,
            loss_function_config=loss_function_config,
            optimizer_config=optimizer_config,
            augmentations_config=augmentations_config,
            scheduler_config=scheduler_config,
        )

    def setup_metrics(self) -> MetricCollection:
        return MetricCollection(
            [
                MeanAbsoluteError(),
                MeanSquaredError(),
                R2Score(),
                MeanSquaredLogError(),
            ]
        )

    def evaluation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int, stage: str) -> Dict[str, Tensor]:
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
