# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import LightningModule, seed_everything
from numpy import ndarray
from PIL import Image
from torch import Tensor
from torch.nn.parameter import Parameter
from torchvision.transforms.functional import to_pil_image

from src.aml.components.classificator_training.arg_parser import get_config
from src.aml.components.classificator_training.config import ClassificationConfig
from src.aml.components.classificator_training.data import ClassificationDataModule, DropletDrugClassificationDataset
from src.common.consts.directories import ARTIFACTS_DIR
from src.common.consts.extensions import PNG
from src.common.utils.logger import get_logger
from src.machine_learning.classification.module import ClassificationLightningModule
from src.machine_learning.preprocessing.factory import create_preprocessor

_logger = get_logger(__name__)


class ClassActivationMapVisualizer:
    """
    Class to visualize Class Activation Maps (CAM) for image classification models.

    This class provides functionality to generate and visualize CAMs for different layers of
    a convolutional neural network. These maps highlight the image regions most influential
    in the model's classification decision, providing insights into model behavior.

    Args:
        checkpoint_path (Path): Path to the model checkpoint.
        feature_layer (str): The name of the neural network layer to visualize.

    Attrs:
        config (ClassificationConfig): Configuration for the model and data.
        model (ClassificationLightningModule): The trained classification model.
        data_module (ClassificationDataModule): Data module for loading and processing data.
        feature_layer (str): The name of the layer to extract features from for CAM visualization.
        model_dict (Dict[str, Any]): Dictionary mapping layer names to layer modules.
    """

    def __init__(self, config: ClassificationConfig, checkpoint_path: Path, feature_layer: str):
        self.config = config
        self.model = self.load_model(checkpoint_path)
        self.data_module = self.prepare_data()
        self.feature_layer = feature_layer

        # Converting the model into a dictionary for easy access to its layers.
        self.model_dict: Dict[str, Any] = dict([*self.model.model.named_children()])

        self.features: List[Tensor] = []

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

    def prepare_data(self) -> ClassificationDataModule:
        """
        Set up the data module for the classifier.

        Returns:
            ClassificationDataModule: Data module with the configured preprocessor.
        """
        _logger.info("Setting up data module.")
        preprocessor = create_preprocessor(config=self.config.preprocessing) if self.config.preprocessing else None

        dm = ClassificationDataModule(config=self.config.data, preprocessor=preprocessor)
        dm.setup()

        return dm

    def get_sample(self, sample_id: int) -> Tuple[Tensor, Tensor]:
        """
        Retrieve a specific sample from the test dataset.

        Args:
            sample_id (int): Index of the sample to retrieve.

        Returns:
            Tuple[Tensor, Tensor]: Tuple of image and label tensors for the sample.
        """
        _logger.info(f"Retrieving sample with ID: {sample_id}")
        dataset = self.data_module.test_dataset

        if dataset is None:
            raise ValueError("No test dataset available. Please run `setup()` method on the data module first.")

        image, label = dataset[sample_id]

        return image, label

    def register_hook(self) -> None:
        """
        Register a hook to the target layer to capture its features.
        """

        def hook_function(module: LightningModule, input: Tensor, output: Tensor) -> None:
            """
            A hook function that captures the output of a layer during the forward pass.

            Args:
                module (LightningModule): The current module.
                input (Tensor): The input tensor to the module.
                output (Tensor): The output tensor from the module.
            """
            self.features.append(output)

        # Registering the forward hook to the specified layer
        layer_module = dict([*self.model.model.named_modules()])[self.feature_layer]
        layer_module.register_forward_hook(hook_function)

    def extract_features(self, input_image: Tensor) -> Tensor:
        """
        Extract features from a specified layer for a given input image using a forward hook.

        This method dynamically registers a hook on the specified layer to capture output features,
        which are used for generating the Class Activation Map.

        Args:
            input_image (Tensor): The input image tensor.

        Returns:
            Tensor: Extracted feature tensor from the specified layer.

        Raises:
            ValueError: If the specified layer is not found in the model.
            RuntimeError: If no features are extracted from the layer (indicating a possible issue).
        """
        _logger.info(f"Extracting features from layer: {self.feature_layer}")

        if self.feature_layer not in self.model_dict:
            raise ValueError(f"Layer '{self.feature_layer}' not found in model.")

        self.register_hook()

        # Performing a forward pass with the input image to trigger the hook
        _ = self.model(input_image.unsqueeze(0).to(self.model.device))

        if not self.features:
            raise RuntimeError(f"No features extracted from layer: {self.feature_layer}")

        return self.features[0]

    def generate_cam(self, features: Tensor, target_class: Tensor) -> ndarray:
        """
        Generate the Class Activation Map for a specific class based on extracted features.

        The method calculates the CAM by multiplying the target class's weights from the final
        fully connected layer with the output features of the specified convolutional layer.

        Args:
            features (Tensor): Extracted features from the specified convolutional layer.
            target_class (Tensor): The target class index for which the CAM is generated.

        Returns:
            ndarray: The Class Activation Map as a NumPy array.
        """
        _logger.info("Generating class activation map.")

        # Extracting the weights of the target class from the final fully connected layer
        params = list(self.model.model.fc.parameters())
        weight_softmax = Parameter(params[0])

        # Dynamically determining the number of features
        num_features = weight_softmax.shape[1]

        # Calculating the class activation features by matrix multiplication
        class_activation_features = torch.matmul(
            weight_softmax[target_class], features.squeeze(0).view(num_features, -1)
        )

        # Reshaping the activation features
        feature_map_size = features.shape[2]
        cam = class_activation_features.view(feature_map_size, feature_map_size).cpu().data.numpy()
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        return cam

    def apply_heatmap(self, cam: np.ndarray, input_image: Tensor) -> Image:
        """
        Apply a heatmap to the Class Activation Map and overlay it on the original image.

        Args:
            cam (ndarray): The Class Activation Map.
            input_image (Tensor): The original input image tensor.

        Returns:
            Image: A PIL Image with the heatmap overlaid on the original image.
        """
        _logger.info("Applying heatmap to class activation map.")

        cam_heatmap = plt.get_cmap("jet")(cam)[:, :, :3]
        cam_heatmap = np.uint8(255 * cam_heatmap)
        cam_heatmap = Image.fromarray(cam_heatmap)
        cam_heatmap = cam_heatmap.resize((input_image.shape[1], input_image.shape[2]), Image.BICUBIC)

        original_image = to_pil_image(input_image)

        return Image.blend(original_image, cam_heatmap, 0.25)

    def visualize_class_activation_map(self, input_image: Tensor, target_class: Tensor) -> Image:
        """
        Generate and visualize the Class Activation Map (CAM) for a given image and target class.

        Args:
            input_image (Tensor): The input image tensor.
            target_class (Tensor): The target class tensor.

        Returns:
            Image: A PIL Image with the CAM overlaid on the original image.
        """
        _logger.info("Visualizing class activation map.")

        features = self.extract_features(input_image)
        cam = self.generate_cam(features=features, target_class=target_class)
        blended_image = self.apply_heatmap(cam=cam, input_image=input_image)

        return blended_image

    def save_class_activation_map(self, image: Image, sample_id: int, save_dir: Path) -> None:
        """
        Save the generated class activation map blended with the input image to a file.

        Args:
            image (Image): The class activation map image to be saved.
            sample_id (int): The ID of the sample.
            save_dir (Path): The directory where the image should be saved.
        """
        save_path = save_dir / self.feature_layer / f"{sample_id=}{PNG}"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(save_path)
        _logger.info(f"Class activation map saved to {save_path}")

    def run(self, sample_id: int, save_dir: Path) -> None:
        """
        Run the visualization process for a given sample and save the resulting class activation map.

        Args:
            sample_id (int): The ID of the sample to visualize.
            save_dir (Path): The directory to save the generated image.
        """
        image, label = self.get_sample(sample_id=sample_id)
        class_activation_img = self.visualize_class_activation_map(input_image=image, target_class=label)
        self.save_class_activation_map(image=class_activation_img, sample_id=sample_id, save_dir=save_dir)


if __name__ == "__main__":
    experiment_dir = ARTIFACTS_DIR / "droplet-drug-classificator" / "2023-12-05_12-52-50"
    checkpoint_path_ = experiment_dir / "checkpoints" / "epoch=2-val_loss=0.0972.ckpt"
    save_dir_ = experiment_dir / "class_activation_maps"
    sample_id_ = 42
    feature_layer_ = "layer4"

    config = get_config()

    seed_everything(seed=config.seed, workers=True)

    visualizer = ClassActivationMapVisualizer(
        config=config, checkpoint_path=checkpoint_path_, feature_layer=feature_layer_
    )
    visualizer.run(sample_id=sample_id_, save_dir=save_dir_)
