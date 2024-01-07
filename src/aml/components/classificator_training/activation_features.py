# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List, Tuple

import torch
from lightning import LightningModule, seed_everything
from PIL import Image
from torch import Tensor
from torch.optim import Adam
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from src.aml.components.classificator_training.arg_parser import get_config
from src.aml.components.classificator_training.config import ClassificationConfig
from src.aml.components.classificator_training.data import DropletDrugClassificationDataset
from src.common.consts.directories import ARTIFACTS_DIR
from src.common.consts.extensions import PNG
from src.common.utils.logger import get_logger
from src.machine_learning.classification.module import ClassificationLightningModule

_logger = get_logger(__name__)


class ActivationFeatureVisualizer:
    """
    A class to visualize activation features of layers in a PyTorch model.

    This class generates images that maximize the activation of specified layers in a model,
    helping to understand what features the model is looking for in those layers.

    Attrs:
        model (LightningModule): The trained PyTorch Lightning model.
        feature_layer (str): The name of the layer to visualize.
    """

    def __init__(self, config: ClassificationConfig, checkpoint_path: Path, feature_layer: str):
        """
        Initialize the visualizer with a model and a target layer.

        Args:
            config (ClassificationConfig): Configuration for the model and data.
            checkpoint_path (Path): The checkpoint path to the trained model.
            feature_layer (str): The name of the layer to visualize.
        """
        self.config = config
        self.feature_layer = feature_layer
        self.model = self.load_model(checkpoint_path)
        self.activations: List[Tensor] = []

    def load_model(self, checkpoint_path: Path) -> ClassificationLightningModule:
        """
        Load the model from a checkpoint.

        Args:
            checkpoint_path (Path): Path to the model checkpoint.

        Returns:
            ClassificationLightningModule: Loaded model in evaluation mode.
        """
        _logger.info(f"Loading model from checkpoint: {checkpoint_path}")

        model = ClassificationLightningModule.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            classes=DropletDrugClassificationDataset.CLASSES,
            model_config=self.config.model,
            loss_function_config=self.config.loss_function,
            optimizer_config=self.config.optimizer,
            metrics_config=self.config.metrics,
            augmentations_config=self.config.augmentations,
            scheduler_config=self.config.scheduler,
        )
        model.eval()

        return model

    def register_hook(self) -> None:
        """
        Register a hook to the target layer to capture its activations.
        """

        def hook_function(module: LightningModule, input: Tensor, output: Tensor) -> None:
            """
            A hook function that captures the output of a layer during the forward pass.

            Args:
                module (LightningModule): The current module.
                input (Tensor): The input tensor to the module.
                output (Tensor): The output tensor from the module.
            """
            self.activations.append(output)

        layer_module = dict([*self.model.model.named_modules()])[self.feature_layer]
        layer_module.register_forward_hook(hook_function)

    def optimize_image(
        self, learning_rate: float, iterations: int, save_dir: Path, image_size: Tuple[int, int]
    ) -> Tensor:
        """
        Optimize an image to maximize activations of the target layer.

        Args:
            learning_rate (float): The learning rate for the optimization.
            iterations (int): The number of iterations for the optimization process.
            save_dir (Path): The directory to save intermediate images to.
            image_size (Tuple[int, int]): The size of the image to optimize.

        Returns:
            Tensor: The optimized image tensor.
        """
        # Initialization of the image to be optimized
        optimized_image = torch.randn(1, 3, *image_size, device=self.model.device, requires_grad=True)
        optimizer = Adam([optimized_image], lr=learning_rate)

        for step in tqdm(range(iterations), desc="Optimizing image"):
            optimizer.zero_grad()

            # Clear previous activations to save memory
            self.activations.clear()

            # Performing a forward pass with the input image to trigger the hook
            _ = self.model(optimized_image)

            # Loss is negative to maximize activation
            loss = -self.activations[-1].mean()
            loss.backward()

            optimizer.step()

            # Clamp the image data to valid range [0, 1]
            optimized_image.data.clamp_(0, 1)

            if step % 100 == 0:
                self._save_intermediate_image(
                    image_tensor=optimized_image, step=step, save_dir=save_dir, learning_rate=learning_rate
                )

            torch.cuda.empty_cache()

        return optimized_image

    def _save_intermediate_image(self, image_tensor: Tensor, step: int, save_dir: Path, learning_rate: float) -> None:
        """
        Save an intermediate image during the optimization process.

        Args:
            image_tensor (Tensor): The image tensor to save.
            step (int): The current step in the optimization process.
            save_dir (Path): The directory to save the image.
            learning_rate (float): The learning rate used in optimization.
        """
        grad_norm = image_tensor.grad.norm()
        intermediate_image = to_pil_image(image_tensor.detach().squeeze())
        intermediate_image_path = (
            save_dir / f"{self.feature_layer}" / f"{learning_rate=}" / f"{step=}_{grad_norm=:.4f}{PNG}"
        )
        intermediate_image_path.parent.mkdir(parents=True, exist_ok=True)
        intermediate_image.save(intermediate_image_path)
        _logger.info(f"Iteration {step}: Gradient Norm: {grad_norm}")

    def run(
        self,
        image_size: Tuple[int, int],
        learning_rate: float,
        iterations: int,
        save_dir: Path,
    ) -> Image:
        """
        Generate and visualize the activation features.

        Args:
            image_size (Tuple[int, int]): The size of the image to optimize.
            learning_rate (float): The learning rate for the optimization.
            iterations (int): The number of iterations for the optimization process.
            save_dir (Path): The directory to save intermediate images to.

        Returns:
            Image: A PIL Image showing the learned features that maximize the layer's activations.
        """
        self.register_hook()
        self.optimize_image(
            learning_rate=learning_rate, iterations=iterations, save_dir=save_dir, image_size=image_size
        )


if __name__ == "__main__":
    experiment_dir = ARTIFACTS_DIR / "droplet-drug-classificator" / "2023-12-05_12-52-50"
    checkpoint_path_ = experiment_dir / "checkpoints" / "epoch=2-val_loss=0.0972.ckpt"
    save_dir_ = experiment_dir / "activation_visualization"
    target_layer_ = "layer4"
    learning_rate_ = 0.01
    iterations_ = 10000
    image_size_ = (256, 256)

    config = get_config()

    seed_everything(seed=config.seed, workers=True)

    visualizer = ActivationFeatureVisualizer(
        config=config,
        checkpoint_path=checkpoint_path_,
        feature_layer=target_layer_,
    )
    visualizer.run(image_size=image_size_, learning_rate=learning_rate_, iterations=iterations_, save_dir=save_dir_)
