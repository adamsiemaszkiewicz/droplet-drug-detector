# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict, List, Literal, Tuple

import matplotlib.pyplot as plt
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import MLFlowLogger, WandbLogger
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
    LABEL_PAD: int = 10
    CMAP: str = "Blues"
    FLOAT_PRECISION: str = ".2f"

    def plot_confusion_matrix(
        self, cm: Tensor, class_names: List[str], stage: str, epoch: int, output_path: Path
    ) -> None:
        """
        Plots and saves a confusion matrix as a heatmap.

        Args:
            cm (Tensor): The confusion matrix to plot, provided as a Tensor.
            class_names (List[str]): The names corresponding to the classes in the confusion matrix.
            stage (str): The stage of model evaluation ('train', 'val', or 'test') for which the matrix is plotted.
            epoch (int): The epoch number at which the confusion matrix is generated.
            output_path (Path): The path to save the confusion matrix plot.
        """
        plt.figure(figsize=self.FIG_SIZE)

        heatmap(
            cm,
            annot=True,
            fmt=self.FLOAT_PRECISION,
            xticklabels=class_names,
            yticklabels=class_names,
            cmap=self.CMAP,
            annot_kws={"size": self.FONT_SIZE},
            vmin=0,
            vmax=1,
        )

        plt.ylabel("Ground truth", fontsize=self.FONT_SIZE, labelpad=self.LABEL_PAD)
        plt.xlabel("Predictions", fontsize=self.FONT_SIZE, labelpad=self.LABEL_PAD)

        plt.xticks(rotation=45)
        plt.yticks(rotation=45)

        plt.title(f"Confusion Matrix\nstage:{stage}, epoch:{epoch}", fontsize=ConfusionMatrixPlotter.TITLE_SIZE, pad=16)

        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()

        _logger.info(f"Confusion Matrix for {stage} saved to {output_path}")


class ConfusionMatrixCallback(Callback):
    """
    PyTorch Lightning callback that calculates and saves confusion matrices at the end of each epoch of every stage.

    Attributes:
        save_dir (Path): The directory to save output plots.
        num_classes (int): The number of classes for the confusion matrix.
        class_names (List[str]): The names of the classes for plotting.
        task (Literal["binary", "multiclass", "multilabel"]): The type of classification task.
        log_train (bool): Flag to control whether to save the confusion matrix during training.
        log_val (bool): Flag to control whether to save the confusion matrix during validation.
        log_test (bool): Flag to control whether to save the confusion matrix during testing.
        normalize (Literal["true", "pred", "all", "none"]): Normalization of the confusion matrix.
    """

    def __init__(
        self,
        save_dir: Path,
        class_dict: Dict[int, str],
        task: Literal["binary", "multiclass", "multilabel"],
        log_train: bool,
        log_val: bool,
        log_test: bool,
        normalize: Literal["true", "pred", "all", "none"] = "true",
    ) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.class_dict = class_dict
        self.num_classes = len(class_dict)
        self.class_names = list(class_dict.values())
        self.task = task
        self.normalize = normalize
        self.log_train = log_train
        self.log_val = log_val
        self.log_test = log_test

        self.plotter = ConfusionMatrixPlotter()

        self.confusion_matrix_metric = ConfusionMatrix(
            num_classes=self.num_classes, task=self.task, normalize=self.normalize
        )

    def process_epoch_end(self, trainer: Trainer, pl_module: LightningModule, loader_name: str, stage: str) -> None:
        """
        Common logic for processing the end of an epoch.

        Args:
            trainer (Trainer): The Trainer object.
            pl_module (LightningModule): The LightningModule object.
            loader_name (str): Name of the dataloader attribute in trainer.
            stage (str): Stage name (train, val, or test).
        """
        preds, targets = self.get_all_preds_and_targets(trainer=trainer, pl_module=pl_module, loader_name=loader_name)
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

    def get_all_preds_and_targets(
        self, trainer: Trainer, pl_module: LightningModule, loader_name: str
    ) -> Tuple[Tensor, Tensor]:
        """
        Collect all predictions and targets.

        Args:
            trainer (Trainer): The Trainer object.
            pl_module (LightningModule): The LightningModule object.
            loader_name (str): Name of the dataloader attribute in trainer.

        Returns:
            Tuple[Tensor, Tensor]: Tuple containing all predictions and targets.
        """
        all_preds = []
        all_targets = []
        dataloader = getattr(trainer, loader_name)
        for batch in dataloader:
            x, y = batch
            x = x.to(pl_module.device)
            logits = pl_module(x)
            preds = self._compute_predictions(logits=logits)
            all_preds.append(preds)
            all_targets.append(y)
        all_preds = torch.cat(all_preds, dim=0).cpu()
        all_targets = torch.cat(all_targets, dim=0).cpu()

        return all_preds, all_targets

    def _compute_predictions(self, logits: Tensor) -> Tensor:
        """
        Computes predictions from the model's logits based on the specified task type.

        Args:
            logits (Tensor): The logits output by the model.

        Returns:
            Tensor: The computed predictions.
        """
        if self.task in ["binary", "multiclass"]:
            return logits.argmax(dim=1)
        elif self.task == "multilabel":
            return (logits.sigmoid() > 0.5).type(torch.int)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        PyTorch Lightning hook that is called when the train epoch ends.
        """
        if self.log_train:
            self.process_epoch_end(
                trainer=trainer, pl_module=pl_module, loader_name="train_dataloader", stage=STAGE_TRAINING
            )

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        PyTorch Lightning hook that is called when the validation epoch ends.
        """
        if trainer.sanity_checking:
            _logger.info("Skipping confusion matrix saving during sanity check.")
            return

        if self.log_val:
            self.process_epoch_end(
                trainer=trainer, pl_module=pl_module, loader_name="val_dataloaders", stage=STAGE_VALIDATION
            )

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        PyTorch Lightning hook that is called when the test epoch ends.
        """
        if self.log_test:
            self.process_epoch_end(
                trainer=trainer, pl_module=pl_module, loader_name="test_dataloaders", stage=STAGE_TESTING
            )
