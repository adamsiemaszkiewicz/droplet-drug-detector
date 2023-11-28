# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import MLFlowLogger, WandbLogger

from src.common.consts.extensions import PNG
from src.common.consts.machine_learning import STAGE_TESTING, STAGE_TRAINING, STAGE_VALIDATION
from src.common.utils.logger import get_logger

_logger = get_logger(__name__)


class LearningCurvePlotter:
    FIG_SIZE: Tuple[int, int] = (12, 10)
    TITLE_SIZE: int = 14
    TITLE_PADDING: int = 16
    FONT_SIZE: int = 10
    SUBPLOT_ADJUST_BOTTOM: float = 0.15
    SUBPLOT_ADJUST_LEFT: float = 0.15
    STAGE_COLORS: Dict[str, str] = {STAGE_TRAINING: "blue", STAGE_VALIDATION: "green", STAGE_TESTING: "red"}
    PROPS_COLOR: str = "black"
    ARROW_STYLE: str = "->"
    SINGLE_VALUE_MARKER: str = "o"
    LABEL_PAD: int = 10
    NUM_Y_TICKS: int = 10
    FLOAT_PRECISION: str = ".2f"

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def annotate_min_value(self, values: List[float], x_coord: Optional[int] = None) -> None:
        """
        Function to annotate the minimum loss value of a stage.

        Args:
            values (List[float]): List of loss values.
            x_coord (Optional[int]): The x-coordinate for annotation (for single value).
        """
        plt.gca()
        min_value_idx = np.argmin(values) if x_coord is None else x_coord
        min_value = values[min_value_idx] if x_coord is None else values[0]
        plt.annotate(
            text=f"Min: {min_value:{self.FLOAT_PRECISION}}",
            xy=(min_value_idx, min_value),
            xycoords="data",
            xytext=(0, -20),
            textcoords="offset points",
            arrowprops=dict(facecolor=self.PROPS_COLOR, arrowstyle=self.ARROW_STYLE),
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=self.FONT_SIZE,
        )

    def annotate_max_value(self, values: List[float], x_coord: Optional[int] = None) -> None:
        """
        Function to annotate the maximum value of a stage.

        Args:
            values (List[float]): List of values.
            x_coord (Optional[int]): The x-coordinate for annotation (for single value).
        """
        plt.gca()
        max_value_idx = np.argmax(values) if x_coord is None else x_coord
        max_value = values[max_value_idx] if x_coord is None else values[0]
        plt.annotate(
            f"Max: {max_value:{self.FLOAT_PRECISION}}",
            xy=(max_value_idx, max_value),
            xycoords="data",
            xytext=(0, 20),
            textcoords="offset points",
            arrowprops=dict(facecolor=self.PROPS_COLOR, arrowstyle=self.ARROW_STYLE),
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=self.FONT_SIZE,
        )

    def annotate_single_value(self, values: List[float], x_coord: Optional[int] = None) -> None:
        """
        Function to annotate a single value of a stage.

        Args:
            values (List[float]): List of values.
            x_coord (Optional[int]): The x-coordinate for annotation (for single value).
        """
        plt.gca()
        value_idx = x_coord if x_coord is not None else 0
        value = values[0]
        plt.annotate(
            f"Value: {value:{self.FLOAT_PRECISION}}",
            xy=(value_idx, value),
            xycoords="data",
            xytext=(0, 20),
            textcoords="offset points",
            arrowprops=dict(facecolor=self.PROPS_COLOR, arrowstyle=self.ARROW_STYLE),
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=self.FONT_SIZE,
        )

    def plot_learning_curve(self, name: str, metric_values: Dict[str, List[float]], total_epochs: int) -> Path:
        """
        Plot and save the learning curve for a single metric.

        Args:
            name (str): Name of the metric.
            metric_values (Dict[str, List[float]]): Dictionary of metric values per stage.
            total_epochs (int): Total number of epochs.

        Returns:
            Path: Path to the saved plot.
        """
        plt.figure(figsize=self.FIG_SIZE)

        min_value = min(min(values) for values in metric_values.values() if values)
        max_value = max(max(values) for values in metric_values.values() if values)

        for stage, values in metric_values.items():
            if values:
                if len(values) == 1:
                    color = self.STAGE_COLORS.get(stage, "gray")
                    plt.plot(total_epochs, values[0], marker=self.SINGLE_VALUE_MARKER, label=stage, color=color)
                    self.annotate_single_value(values=values, x_coord=total_epochs)
                else:
                    color = self.STAGE_COLORS.get(stage, "gray")
                    plt.plot(values, label=stage, color=color)
                    self.annotate_min_value(values=values)
                    self.annotate_max_value(values=values)

        # Custom y-axis label and ticks for loss
        if name.lower() == "loss":
            plt.yscale("log")
            plt.ylabel(ylabel="Loss (logarithmic scale)", fontsize=self.FONT_SIZE, labelpad=self.LABEL_PAD)

            y_ticks = np.geomspace(min_value, max_value, num=self.NUM_Y_TICKS)
            plt.yticks(y_ticks, [f"{tick:{self.FLOAT_PRECISION}}" for tick in y_ticks])
        else:
            y_ticks = np.linspace(min_value, max_value, num=self.NUM_Y_TICKS)
            plt.yticks(y_ticks, [f"{tick:{self.FLOAT_PRECISION}}" for tick in y_ticks])
            plt.ylabel(ylabel=name, fontsize=self.FONT_SIZE, labelpad=self.LABEL_PAD)

        plt.xlabel("Epoch number", fontsize=self.FONT_SIZE, labelpad=self.LABEL_PAD)
        x_ticks = list(range(total_epochs + 1))
        plt.xticks(ticks=x_ticks)

        plt.legend(title=f"average {name} per epoch", fontsize=self.FONT_SIZE)

        plt.grid(True)

        plt.title(f"Learning Curve ({name})", fontsize=self.TITLE_SIZE, pad=self.TITLE_PADDING)

        plt.subplots_adjust(bottom=0.15, left=0.15)
        plt.tight_layout()

        output_path = self.output_dir / f"learning_curve_{name}{PNG}"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            _logger.info(f"{name} curve already exists at {output_path}. Skipping...")
            return output_path

        plt.savefig(output_path)
        plt.close()

        _logger.info(f"{name} curve saved to {output_path}")

        return output_path


class LearningCurveCallback(Callback):
    """
    PyTorch Lightning callback that saves the learning curve plot for training,
    validation, and test losses, as well as other specified metrics at the end of each epoch.

    Args:
        output_dir (Path): The directory to save output plots.
        log_loss (bool): Whether to log the loss learning curve.
        log_metrics (bool): Whether to log the metrics learning curve.

    Attrs:
        output_dir (Path): The directory to save output plots.
        log_loss (bool): Whether to log the loss learning curve.
        log_metrics (bool): Whether to log the metrics learning curve.
        plotting_utility (LearningCurvePlottingUtility): The plotting utility.
        epoch_losses (Dict[str, List[float]]): Recorded loss for each training stage per epoch.
        epoch_metrics (Dict[str, Dict[str, List[float]]]): Recorded metrics for each training stage per epoch.
    """

    def __init__(self, output_dir: Path, log_loss: bool, log_metrics: bool) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.log_loss = log_loss
        self.log_metrics = log_metrics

        self.plotter = LearningCurvePlotter(output_dir=output_dir)

        self.epoch_losses: Dict[str, List[float]] = {STAGE_TRAINING: [], STAGE_VALIDATION: [], STAGE_TESTING: []}
        self.epoch_metrics: Dict[str, Dict[str, List[float]]] = {
            STAGE_TRAINING: {},
            STAGE_VALIDATION: {},
            STAGE_TESTING: {},
        }

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        PyTorch Lightning hook that is called when the train epoch ends.
        """
        self.update_metrics_and_losses(trainer=trainer, stage=STAGE_TRAINING)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        PyTorch Lightning hook that is called when the validation epoch ends.
        """
        self.update_metrics_and_losses(trainer=trainer, stage=STAGE_VALIDATION)

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        PyTorch Lightning hook that is called when the test epoch ends.
        """
        self.update_metrics_and_losses(trainer=trainer, stage=STAGE_TESTING)
        self.plot_learning_curves(trainer=trainer)

    def update_metrics_and_losses(self, trainer: Trainer, stage: str) -> None:
        """
        Update epoch losses and metrics for a given stage.

        Args:
            trainer (Trainer): The Trainer object.
            stage (str): The stage name (training, validation, or test).
        """
        loss = trainer.callback_metrics.get(f"{stage}_loss")
        if loss is not None:
            self.epoch_losses[stage].append(loss.item())

        for key, value in trainer.callback_metrics.items():
            if key.startswith(stage) and not (key.endswith("_epoch") or key.endswith("_step")):
                metric_name = key[len(stage) + 1 :]
                if metric_name == "loss":
                    continue
                self.epoch_metrics[stage].setdefault(metric_name, []).append(value.item())

    def plot_learning_curves(self, trainer: Trainer) -> None:
        """
        Plot all learning curves

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer object.
        """
        if self.log_loss:
            self._plot_and_log_loss_curve(trainer)

        if self.log_metrics:
            self._plot_and_log_metrics_curves(trainer)

    def _plot_and_log_loss_curve(self, trainer: Trainer) -> None:
        """
        Plot and log the loss curve.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer object.
        """
        loss_values = {
            STAGE_TRAINING: self.epoch_losses[STAGE_TRAINING],
            STAGE_VALIDATION: self.epoch_losses[STAGE_VALIDATION],
            STAGE_TESTING: self.epoch_losses[STAGE_TESTING],
        }

        total_epochs = len(self.epoch_losses[STAGE_TRAINING])
        plot_path = self.plotter.plot_learning_curve(name="loss", metric_values=loss_values, total_epochs=total_epochs)
        self._log_curve(plot_path=plot_path, metric_name="loss", trainer=trainer)

    def _plot_and_log_metrics_curves(self, trainer: Trainer) -> None:
        """
        Plot and log the metrics curves.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer object.
        """
        for metric_name, _ in self.epoch_metrics[STAGE_TRAINING].items():
            metric_values = {
                STAGE_TRAINING: self.epoch_metrics[STAGE_TRAINING][metric_name],
                STAGE_VALIDATION: self.epoch_metrics[STAGE_VALIDATION].get(metric_name, []),
                STAGE_TESTING: self.epoch_metrics[STAGE_TESTING].get(metric_name, []),
            }

            total_epochs = len(self.epoch_metrics[STAGE_TRAINING][metric_name])
            plot_path = self.plotter.plot_learning_curve(
                name=metric_name, metric_values=metric_values, total_epochs=total_epochs
            )
            self._log_curve(plot_path=plot_path, metric_name=metric_name, trainer=trainer)

    def _log_curve(self, plot_path: Path, metric_name: str, trainer: Trainer) -> None:
        """
        Log the curve if the appropriate logger is available.

        Args:
            plot_path (Path): Path to the saved plot.
            metric_name (str): Name of the metric.
            trainer (Trainer): The PyTorch Lightning Trainer object.
        """
        for logger in trainer.loggers:
            if isinstance(logger, MLFlowLogger):
                _logger.info(f"Logging {metric_name} learning curve to MLFlow.")
                run = logger.experiment
                run.log_artifact(run_id=logger.run_id, local_path=plot_path.as_posix(), artifact_path="learning_curve")
            if isinstance(logger, WandbLogger):
                _logger.info(f"Logging {metric_name} learning curve to Wandb.")
                run = logger.experiment
                run.log_artifact(plot_path.as_posix())
