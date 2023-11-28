# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict, List, Literal, Tuple

import matplotlib.pyplot as plt
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import Logger, MLFlowLogger
from seaborn import heatmap
from torch import Tensor
from torchmetrics import ConfusionMatrix

from src.common.consts.machine_learning import STAGE_TESTING, STAGE_TRAINING, STAGE_VALIDATION
from src.common.utils.logger import get_logger

_logger = get_logger(__name__)


class ConfusionMatrixCallback(Callback):
    """
    PyTorch Lightning callback that calculates and saves confusion matrices at the end of each epoch of every stage.

    Attributes:
        output_dir (Path): The directory to save output plots.
        num_classes (int): The number of classes for the confusion matrix.
        class_names (List[str]): The names of the classes for plotting.
        task (Literal["binary", "multiclass", "multilabel"]): The type of classification task.
        save_train (bool): Flag to control whether to save the confusion matrix during training.
        save_val (bool): Flag to control whether to save the confusion matrix during validation.
        save_test (bool): Flag to control whether to save the confusion matrix during testing.
        normalize (Literal["true", "pred", "all", "none"]): Normalization of the confusion matrix.
    """

    def __init__(
        self,
        output_dir: Path,
        class_dict: Dict[int, str],
        task: Literal["binary", "multiclass", "multilabel"],
        save_train: bool,
        save_val: bool,
        save_test: bool,
        normalize: Literal["true", "pred", "all", "none"] = "true",
    ) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.class_dict = class_dict
        self.num_classes = len(class_dict)
        self.class_names = list(class_dict.values())
        self.task = task
        self.normalize = normalize
        self.save_train = save_train
        self.save_val = save_val
        self.save_test = save_test

        self.confusion_matrix_metric = ConfusionMatrix(
            num_classes=self.num_classes, task=self.task, normalize=self.normalize
        )

    def plot_confusion_matrix(
        self, output_path: Path, cm: Tensor, class_names: List[str], stage: str, epoch: int
    ) -> Path:
        """
        Plots and saves a confusion matrix as a heatmap.

        Args:
            output_path (Path): The path to save the confusion matrix plot.
            cm (Tensor): The confusion matrix to plot, provided as a Tensor.
            class_names (List[str]): The names corresponding to the classes in the confusion matrix.
            stage (str): The stage of model evaluation ('train', 'val', or 'test') for which the matrix is plotted.
            epoch (int): The epoch number at which the confusion matrix is generated.

        Returns:
            Path: The path to the saved plot.
        """
        plt.figure(figsize=(12, 10))
        fontsize = 10

        heatmap(
            cm,
            annot=True,
            fmt=".2f",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues",
            annot_kws={"size": fontsize},
            vmin=0,
            vmax=1,
        )

        plt.ylabel("Ground truth", fontsize=fontsize, labelpad=10)
        plt.xlabel("Predictions", fontsize=fontsize, labelpad=10)

        plt.xticks(rotation=45)
        plt.yticks(rotation=45)

        plt.title(f"Confusion Matrix\nstage:{stage}, epoch:{epoch}", fontsize=14, pad=16)

        plt.subplots_adjust(bottom=0.15, left=0.15)
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(output_path)
        plt.close()

        _logger.info(f"Confusion Matrix for {stage} saved to {output_path}")

        return output_path

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

        output_path = self.output_dir / "confusion_matrix" / stage / f"confusion_matrix_{trainer.current_epoch}.png"

        self.plot_confusion_matrix(
            output_path=output_path, cm=cm, class_names=self.class_names, stage=stage, epoch=trainer.current_epoch
        )

        logger = trainer.logger
        if not isinstance(logger, Logger):
            raise ValueError("Provided logger is not a PyTorch Lightning Logger")
        if isinstance(trainer.logger, MLFlowLogger):
            _logger.info(f"Logging {stage} confusion matrix to MLFlow.")
            logger.experiment.log_artifact(
                run_id=logger.run_id,
                local_path=output_path.as_posix(),
                artifact_path=f"confusion_matrix/{stage}/{trainer.current_epoch}",
            )
        else:
            _logger.info("MLFlow logger not found. Skipping logging of confusion matrix.")

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
        if self.task in ["binary", "multiclass"]:
            return logits.argmax(dim=1)
        elif self.task == "multilabel":
            return (logits.sigmoid() > 0.5).type(torch.int)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        PyTorch Lightning hook that is called when the train epoch ends.
        """
        if self.save_train:
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

        if self.save_val:
            self.process_epoch_end(
                trainer=trainer, pl_module=pl_module, loader_name="val_dataloaders", stage=STAGE_VALIDATION
            )

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        PyTorch Lightning hook that is called when the test epoch ends.
        """
        if self.save_test:
            self.process_epoch_end(
                trainer=trainer, pl_module=pl_module, loader_name="test_dataloaders", stage=STAGE_TESTING
            )
