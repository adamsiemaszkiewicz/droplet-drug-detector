# -*- coding: utf-8 -*-
import heapq
from pathlib import Path
from typing import Any, Dict, List, Tuple

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor
from torchvision.utils import save_image

from src.common.consts.extensions import PNG
from src.common.consts.machine_learning import STAGE_TESTING, STAGE_TRAINING, STAGE_VALIDATION


class MisclassificationCallback(Callback):
    """
    A callback to save top loss images of misclassified samples
    during training, validation, or testing in a PyTorch Lightning training loop.

    Attrs:
        save_dir (Path): Directory path where misclassified images will be saved.
        log_train (bool): Flag to enable logging misclassifications during training stage.
        log_val (bool): Flag to enable logging misclassifications during validation stage.
        log_test (bool): Flag to enable logging misclassifications during testing stage.
        top_n (int): Number of top highest loss misclassified samples to save.
        misclass_heaps (Dict[str, List]): Heaps to store top misclassified samples for each stage.
    """

    def __init__(self, save_dir: Path, log_train: bool, log_val: bool, log_test: bool, top_n: int) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.log_train = log_train
        self.log_val = log_val
        self.log_test = log_test
        self.top_n = top_n
        self.misclass_heaps: Dict[str, List[Any]] = {
            STAGE_TRAINING: [],
            STAGE_VALIDATION: [],
            STAGE_TESTING: [],
        }

    def on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict[str, Tensor],
        batch: Tuple[Tensor, Tensor],
        stage: str,
    ) -> None:
        if self.misclass_heaps[stage] is not None:
            images, labels = batch
            preds = outputs["preds"]
            losses = outputs["per_sample_losses"]

            for idx in range(len(images)):
                if preds[idx] != labels[idx]:
                    loss = losses[idx].item()
                    heapq.heappush(self.misclass_heaps[stage], (-loss, (images[idx], labels[idx], preds[idx])))
                    if len(self.misclass_heaps[stage]) > self.top_n:
                        heapq.heappop(self.misclass_heaps[stage])

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict[str, Tensor],
        batch: Tuple[Tensor, Tensor],
        batch_idx: int,
    ) -> None:
        self.on_batch_end(trainer=trainer, pl_module=pl_module, outputs=outputs, batch=batch, stage=STAGE_TRAINING)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict[str, Tensor],
        batch: Tuple[Tensor, Tensor],
        batch_idx: int,
    ) -> None:
        self.on_batch_end(trainer=trainer, pl_module=pl_module, outputs=outputs, batch=batch, stage=STAGE_VALIDATION)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict[str, Tensor],
        batch: Tuple[Tensor, Tensor],
        batch_idx: int,
    ) -> None:
        self.on_batch_end(trainer=trainer, pl_module=pl_module, outputs=outputs, batch=batch, stage=STAGE_TESTING)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.log_train:
            self._process_heap_and_save_images(trainer=trainer, pl_module=pl_module, stage=STAGE_TRAINING)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.log_val:
            self._process_heap_and_save_images(trainer=trainer, pl_module=pl_module, stage=STAGE_VALIDATION)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.log_test:
            self._process_heap_and_save_images(trainer=trainer, pl_module=pl_module, stage=STAGE_TESTING)

    def _process_heap_and_save_images(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """
        Processes the heap for the given stage and saves images of the top misclassified samples.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer instance.
            pl_module (LightningModule): The LightningModule being trained.
            stage (str): The current training stage (training, validation, or testing).
        """
        heap = self.misclass_heaps[stage]
        if heap:
            while heap:
                loss, (img, true_label, pred_label) = heapq.heappop(heap)
                pred_label = pl_module.classes[pred_label.item()]
                true_label = pl_module.classes[true_label.item()]
                epoch = trainer.current_epoch

                save_path = (
                    self.save_dir / stage / f"{epoch=}" / f"{true_label=}" / f"{pred_label=}" / f"loss={-loss:.4f}{PNG}"
                )
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_image(tensor=img, fp=save_path)
            self.misclass_heaps[stage].clear()
