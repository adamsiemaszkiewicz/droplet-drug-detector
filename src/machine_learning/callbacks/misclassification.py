# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Dict, Tuple

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor
from torchvision.utils import save_image

from src.common.consts.extensions import PNG
from src.common.consts.machine_learning import STAGE_TESTING, STAGE_TRAINING, STAGE_VALIDATION


class MisclassificationCallback(Callback):
    def __init__(self, save_dir: Path, log_train: bool, log_val: bool, log_test: bool):
        super().__init__()
        self.save_dir = save_dir
        self.log_train = log_train
        self.log_val = log_val
        self.log_test = log_test

    def log_misclassifications(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict[str, Tensor],
        batch: Tuple[Tensor, Tensor],
        batch_idx: int,
        stage: str,
    ) -> None:
        images, labels = batch
        preds = outputs["preds"]
        misclassified_indices = torch.where(preds != labels)[0]

        epoch = trainer.current_epoch

        for idx in misclassified_indices:
            img = images[idx]
            pred_label = pl_module.classes[preds[idx].item()]
            true_label = pl_module.classes[labels[idx].item()]

            save_path = (
                self.save_dir
                / stage
                / f"{epoch=}"
                / f"{true_label=}"
                / f"{pred_label=}"
                / f"{batch_idx=}"
                / f"{idx}{PNG}"
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)

            save_image(tensor=img, fp=save_path)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict[str, Tensor],
        batch: Tuple[Tensor, Tensor],
        batch_idx: int,
    ) -> None:
        if self.log_train:
            self.log_misclassifications(
                trainer=trainer,
                pl_module=pl_module,
                outputs=outputs,
                batch=batch,
                batch_idx=batch_idx,
                stage=STAGE_TRAINING,
            )

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict[str, Tensor],
        batch: Tuple[Tensor, Tensor],
        batch_idx: int,
    ) -> None:
        if self.log_val:
            self.log_misclassifications(
                trainer=trainer,
                pl_module=pl_module,
                outputs=outputs,
                batch=batch,
                batch_idx=batch_idx,
                stage=STAGE_VALIDATION,
            )

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict[str, Tensor],
        batch: Tuple[Tensor, Tensor],
        batch_idx: int,
    ) -> None:
        if self.log_test:
            self.log_misclassifications(
                trainer=trainer,
                pl_module=pl_module,
                outputs=outputs,
                batch=batch,
                batch_idx=batch_idx,
                stage=STAGE_TESTING,
            )
