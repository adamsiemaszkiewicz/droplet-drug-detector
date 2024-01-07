# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import MLFlowLogger, WandbLogger
from numpy import ndarray
from seaborn import heatmap
from torch import Tensor
from torchmetrics import ConfusionMatrix

from src.common.consts.machine_learning import STAGE_TESTING, STAGE_TRAINING, STAGE_VALIDATION
from src.common.consts.project import CONFUSION_MATRIX_FOLDER_NAME
from src.common.utils.logger import get_logger

_logger = get_logger(__name__)


class ConfusionMatrixPlotter:
    FIG_SIZE: Tuple[int, int] = (12, 10)
    FONT_SIZE: int = 10
    TITLE_SIZE: int = 14
    TICK_ROTATION: int = 45
    LABEL_PAD: int = 10
    CMAP: str = "Blues"
    FLOAT_PRECISION: str = ".2f"

    def plot_confusion_matrix(
        self, cm: ndarray, class_names: List[str], stage: str, epoch: int, output_path: Path
    ) -> None:
        """
        Plots and saves a confusion matrix as a heatmap.

        Args:
            cm (ndarray): The confusion matrix to plot, provided as a Tensor.
            class_names (List[str]): The names corresponding to the classes in the confusion matrix.
            stage (str): The stage of model evaluation ('train', 'val', or 'test') for which the matrix is plotted.
            epoch (int): The epoch number at which the confusion matrix is generated.
            output_path (Path): The path to save the confusion matrix plot.
        """
        plt.figure(figsize=self.FIG_SIZE)

        cm_normalized = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
        labels = np.asarray(
            [f"{normalized:.2f}\n({actual})" for normalized, actual in zip(cm_normalized.flatten(), cm.flatten())]
        ).reshape(cm.shape)

        heatmap(
            data=cm_normalized,
            annot=labels,
            fmt="",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap=self.CMAP,
            annot_kws={"size": self.FONT_SIZE},
            vmin=0,
            vmax=1,
        )

        plt.ylabel("Ground truth", fontsize=self.FONT_SIZE, labelpad=self.LABEL_PAD)
        plt.xlabel("Predictions", fontsize=self.FONT_SIZE, labelpad=self.LABEL_PAD)

        plt.xticks(rotation=self.TICK_ROTATION)
        plt.yticks(rotation=self.TICK_ROTATION)

        plt.title(f"Confusion Matrix\nstage:{stage}, epoch:{epoch}", fontsize=ConfusionMatrixPlotter.TITLE_SIZE, pad=16)

        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()

        _logger.info(f"Confusion Matrix for {stage} saved to {output_path}")


class ConfusionMatrixCallback(Callback):
    """
    PyTorch Lightning callback that calculates and saves confusion matrices at the end of each epoch of every stage.

    Attrs:
        save_dir (Path): The directory to save output plots.
        num_classes (int): The number of classes for the confusion matrix.
        class_names (List[str]): The names of the classes for plotting.
        task (Literal["binary", "multiclass", "multilabel"]): The type of classification task.
        log_train (bool): Flag to control whether to save the confusion matrix during training.
        log_val (bool): Flag to control whether to save the confusion matrix during validation.
        log_test (bool): Flag to control whether to save the confusion matrix during testing.
    """

    def __init__(
        self,
        save_dir: Path,
        class_dict: Dict[int, str],
        task: Literal["binary", "multiclass", "multilabel"],
        log_train: bool,
        log_val: bool,
        log_test: bool,
    ) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.class_dict = class_dict
        self.num_classes = len(class_dict)
        self.class_names = list(class_dict.values())
        self.task = task
        self.log_train = log_train
        self.log_val = log_val
        self.log_test = log_test

        self.plotter = ConfusionMatrixPlotter()

        self.confusion_matrix_metric = ConfusionMatrix(num_classes=self.num_classes, task=self.task, normalize="none")

        self.tmp_predictions: Dict[str, List[Tensor]] = {
            STAGE_TRAINING: [],
            STAGE_VALIDATION: [],
            STAGE_TESTING: [],
        }
        self.tmp_targets: Dict[str, List[Tensor]] = {
            STAGE_TRAINING: [],
            STAGE_VALIDATION: [],
            STAGE_TESTING: [],
        }

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict[str, Tensor],
        batch: Tuple[Tensor, Tensor],
        batch_idx: int,
    ) -> None:
        self.tmp_predictions[STAGE_TRAINING].append(outputs["preds"])
        self.tmp_targets[STAGE_TRAINING].append(outputs["targets"])

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict[str, Tensor],
        batch: Tuple[Tensor, Tensor],
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.tmp_predictions[STAGE_VALIDATION].append(outputs["preds"])
        self.tmp_targets[STAGE_VALIDATION].append(outputs["targets"])

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict[str, Tensor],
        batch: Tuple[Tensor, Tensor],
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.tmp_predictions[STAGE_TESTING].append(outputs["preds"])
        self.tmp_targets[STAGE_TESTING].append(outputs["targets"])

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        PyTorch Lightning hook that is called when the train epoch ends.
        """
        if self.log_train:
            self._process_epoch_end(trainer=trainer, pl_module=pl_module, stage=STAGE_TRAINING)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        PyTorch Lightning hook that is called when the validation epoch ends.
        """
        if trainer.sanity_checking:
            _logger.info("Skipping confusion matrix saving during sanity check.")
            return

        if self.log_val:
            self._process_epoch_end(trainer=trainer, pl_module=pl_module, stage=STAGE_VALIDATION)

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        PyTorch Lightning hook that is called when the test epoch ends.
        """
        if self.log_test:
            self._process_epoch_end(trainer=trainer, pl_module=pl_module, stage=STAGE_TESTING)

    def _process_epoch_end(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """
        Common logic for processing the end of an epoch.

        Args:
            trainer (Trainer): The Trainer object
            pl_module (LightningModule): The LightningModule object.
            stage (str): Stage name (train, val, or test).
        """
        preds, targets = torch.cat(self.tmp_predictions[stage], dim=0), torch.cat(self.tmp_targets[stage], dim=0)
        self.tmp_predictions[stage], self.tmp_targets[stage] = [], []

        # Move everything to the model's device
        device = pl_module.device
        self.confusion_matrix_metric.to(device)
        preds, targets = preds.to(device), targets.to(device)

        self.confusion_matrix_metric(preds, targets)
        cm = self.confusion_matrix_metric.compute().cpu().numpy()
        self.confusion_matrix_metric.reset()

        plot_path = (
            self.save_dir / CONFUSION_MATRIX_FOLDER_NAME / stage / f"confusion_matrix_{trainer.current_epoch}.png"
        )

        self.plotter.plot_confusion_matrix(
            output_path=plot_path, cm=cm, class_names=self.class_names, stage=stage, epoch=trainer.current_epoch
        )
        self._log_confusion_matrix(trainer=trainer, output_path=plot_path, stage=stage)

    def _log_confusion_matrix(self, trainer: Trainer, output_path: Path, stage: str) -> None:
        """
        Logs the confusion matrix to the specified logger.

        Args:
            trainer (Trainer): The Trainer object from PyTorch Lightning.
            output_path (Path): Path to the saved confusion matrix plot.
            stage (str): The stage of training (e.g., 'train', 'val', 'test').
        """
        for logger in trainer.loggers:
            if isinstance(logger, MLFlowLogger):
                _logger.info(f"Logging {stage} confusion matrix to MLFlow.")
                run = logger.experiment
                run.log_artifact(
                    run_id=logger.run_id, local_path=output_path.as_posix(), artifact_path="learning_curve"
                )
            if isinstance(logger, WandbLogger):
                _logger.info(f"Logging {stage} confusion matrix to Wandb.")
                run = logger.experiment
                run.log_artifact(output_path.as_posix())
