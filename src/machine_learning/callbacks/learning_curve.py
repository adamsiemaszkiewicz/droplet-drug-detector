# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import MLFlowLogger, WandbLogger

from src.common.consts.extensions import PNG
from src.common.consts.machine_learning import STAGE_COLORS, STAGE_TESTING, STAGE_TRAINING, STAGE_VALIDATION
from src.common.utils.logger import get_logger

_logger = get_logger(__name__)


class LearningCurvePlottingUtility:
    FIG_SIZE = (12, 10)
    TITLE_SIZE = 14
    FONT_SIZE = 10
    SUBPLOT_ADJUST_BOTTOM = 0.15
    SUBPLOT_ADJUST_LEFT = 0.15
    STAGE_COLORS = {STAGE_TRAINING: "blue", STAGE_VALIDATION: "green", STAGE_TESTING: "red"}
    PROPS_COLOR = "black"
    ARROW_STYLE = "->"
    SINGLE_VALUE_MARKER = "o"
    LABEL_PAD = 10

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
            f"Min: {min_value:.2f}",
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
            f"Max: {max_value:.2f}",
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
            f"Value: {value:.2f}",
            xy=(value_idx, value),
            xycoords="data",
            xytext=(0, 20),
            textcoords="offset points",
            arrowprops=dict(facecolor=self.PROPS_COLOR, arrowstyle=self.ARROW_STYLE),
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=self.FONT_SIZE,
        )

    def plot_metric_curve(self, name: str, metric_values: Dict[str, List[float]], total_epochs: int) -> None:
        """
        Plot and save the learning curve for a single metric.

        Args:
            name (str): Name of the metric.
            metric_values (Dict[str, List[float]]): Dictionary of metric values per stage.
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

            y_ticks = np.geomspace(min_value, max_value, 10)
            plt.yticks(y_ticks, [f"{tick:.1f}" for tick in y_ticks])
        else:
            y_ticks = np.linspace(min_value, max_value, 10)
            plt.yticks(y_ticks, [f"{tick:.1f}" for tick in y_ticks])
            plt.ylabel(y_label=name, fontsize=self.FONT_SIZE, labelpad=self.LABEL_PAD)

        plt.xlabel("Epoch number", fontsize=self.FONT_SIZE, labelpad=self.LABEL_PAD)
        x_ticks = list(range(total_epochs + 1))
        plt.xticks(ticks=x_ticks)

        plt.legend(title=f"average {name} per epoch", fontsize=self.FONT_SIZE)

        plt.grid(True)

        plt.title(f"Learning Curve ({name})", fontsize=self.TITLE_SIZE, pad=16)

        plt.subplots_adjust(bottom=0.15, left=0.15)
        plt.tight_layout()


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
        epoch_losses (Dict[str, List[float]]): Recorded loss for each training stage per epoch.
        epoch_metrics (Dict[str, Dict[str, List[float]]]): Recorded metrics for each training stage per epoch.
    """

    def __init__(self, output_dir: Path, log_loss: bool, log_metrics: bool) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.log_loss = log_loss
        self.log_metrics = log_metrics

        self.plotting_utility = LearningCurvePlottingUtility()

        self.epoch_losses: Dict[str, List[float]] = {STAGE_TRAINING: [], STAGE_VALIDATION: [], STAGE_TESTING: []}
        self.epoch_metrics: Dict[str, Dict[str, List[float]]] = {
            STAGE_TRAINING: {},
            STAGE_VALIDATION: {},
            STAGE_TESTING: {},
        }

    def _annotate_min_value(self, values: List[float], x_coord: Optional[int] = None) -> None:
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
            f"Min: {min_value:.2f}",
            xy=(min_value_idx, min_value),
            xycoords="data",
            xytext=(0, -20),
            textcoords="offset points",
            arrowprops=dict(facecolor="black", arrowstyle="->"),
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=10,
        )

    def _annotate_max_value(self, values: List[float], x_coord: Optional[int] = None) -> None:
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
            f"Max: {max_value:.2f}",
            xy=(max_value_idx, max_value),
            xycoords="data",
            xytext=(0, 20),
            textcoords="offset points",
            arrowprops=dict(facecolor="black", arrowstyle="->"),
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=10,
        )

    def _annotate_single_value(self, values: List[float], x_coord: Optional[int] = None) -> None:
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
            f"Value: {value:.2f}",
            xy=(value_idx, value),
            xycoords="data",
            xytext=(0, 20),
            textcoords="offset points",
            arrowprops=dict(facecolor="black", arrowstyle="->"),
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=10,
        )

    def plot_metric_curve(self, name: str, metric_values: Dict[str, List[float]]) -> Path:
        """
        Plot and save the learning curve for a single metric.

        Args:
            name (str): Name of the metric.
            metric_values (Dict[str, List[float]]): Dictionary of metric values per stage.

        Returns:
            Path: The path to the saved plot.
        """
        plt.figure(figsize=(12, 10))
        fontsize = 10

        total_epochs = len(self.epoch_losses[STAGE_TRAINING])

        min_value = min(min(values) for values in metric_values.values() if values)
        max_value = max(max(values) for values in metric_values.values() if values)

        for stage, values in metric_values.items():
            if values:
                if len(values) == 1:
                    marker = "o" if stage == STAGE_TRAINING else "s" if stage == STAGE_VALIDATION else "d"
                    color = STAGE_COLORS.get(stage, "gray")
                    plt.plot(total_epochs, values[0], marker, label=stage, color=color)
                    self._annotate_single_value(values=values, x_coord=total_epochs)
                else:
                    color = STAGE_COLORS.get(stage, "gray")
                    plt.plot(values, label=stage, color=color)
                    self._annotate_min_value(values=values)
                    self._annotate_max_value(values=values)

        # Custom y-axis label and ticks for loss
        if name.lower() == "loss":
            plt.yscale("log")
            plt.ylabel("Loss (logarithmic scale)", fontsize=fontsize, labelpad=10)

            y_ticks = np.geomspace(min_value, max_value, 10)
            plt.yticks(y_ticks, [f"{tick:.1f}" for tick in y_ticks])
        else:
            y_ticks = np.linspace(min_value, max_value, 10)
            plt.yticks(y_ticks, [f"{tick:.1f}" for tick in y_ticks])
            plt.ylabel(name, fontsize=fontsize, labelpad=10)

        plt.xlabel("Epoch number", fontsize=fontsize, labelpad=10)
        x_ticks = list(range(total_epochs + 1))
        plt.xticks(ticks=x_ticks)

        plt.legend(title=f"average {name} per epoch", fontsize=fontsize)

        plt.grid(True)

        plt.title(f"Learning Curve ({name})", fontsize=14, pad=16)

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

        plot_path = self.plot_metric_curve(name="loss", metric_values=loss_values)
        self._log_curve(plot_path, "loss", trainer)

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

            plot_path = self.plot_metric_curve(name=metric_name, metric_values=metric_values)
            self._log_curve(plot_path, metric_name, trainer)

    def _log_curve(self, plot_path: Path, curve_name: str, trainer: Trainer) -> None:
        """
        Log the curve if the appropriate logger is available.

        Args:
            plot_path (Path): Path to the saved plot.
            curve_name (str): Name of the curve.
            trainer (Trainer): The PyTorch Lightning Trainer object.
        """
        if not plot_path.exists():
            _logger.info(f"{curve_name} curve saved to {plot_path}")
            logger = trainer.logger
            if isinstance(logger, MLFlowLogger):
                _logger.info(f"Logging {curve_name} learning curve to MLFlow.")
                logger.experiment.log_artifact(
                    run_id=logger.run_id, local_path=plot_path.as_posix(), artifact_path="learning_curve"
                )
            elif isinstance(logger, WandbLogger):
                _logger.info(f"Logging {curve_name} learning curve to Wandb.")
                logger.experiment.log_image(images=[plot_path.as_posix()])
            else:
                _logger.info(f"Logger type for {curve_name} curve logging not supported.")
        else:
            _logger.info(f"{curve_name} curve already exists at {plot_path}. Skipping...")

    def _update_metrics_and_losses(self, trainer: Trainer, stage: str) -> None:
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
                self.epoch_metrics[stage].setdefault(metric_name, []).append(value.item())

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        PyTorch Lightning hook that is called when the train epoch ends.
        """
        self._update_metrics_and_losses(trainer=trainer, stage=STAGE_TRAINING)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        PyTorch Lightning hook that is called when the validation epoch ends.
        """
        self._update_metrics_and_losses(trainer=trainer, stage=STAGE_VALIDATION)

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        PyTorch Lightning hook that is called when the test epoch ends.
        """
        self._update_metrics_and_losses(trainer=trainer, stage=STAGE_TESTING)
        self.plot_learning_curves(trainer=trainer)
