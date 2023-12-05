# -*- coding: utf-8 -*-
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import LightningModule, seed_everything
from PIL import Image
from torch import Tensor
from torchvision.transforms.functional import to_pil_image

from src.aml.components.classificator_training.arg_parser import get_config
from src.aml.components.classificator_training.config import ClassificationConfig
from src.aml.components.classificator_training.data import ClassificationDataModule, DropletDrugClassificationDataset
from src.common.consts.directories import ARTIFACTS_DIR
from src.common.utils.logger import get_logger
from src.machine_learning.classification.module import ClassificationLightningModule
from src.machine_learning.preprocessing.factory import create_preprocessor

_logger = get_logger(__name__)


def visualize_class_activation_map(model: LightningModule, input_image: Tensor, target_class: Tensor) -> Image:
    """
    Visualize the Class Activation Map (CAM) for a given model and input image.

    This function generates a CAM for the specified target class, using the last convolutional layer of the model.
    The CAM is then overlaid on the original image, highlighting the regions most relevant for the model's prediction.

    Args:
        model (Module): The trained model, expected to be a PyTorch model.
        input_image (Tensor): The input image tensor, normalized and in the format expected by the model.
        target_class (int): The target class index for which CAM is to be visualized.

    Returns:
        Image: A PIL Image with the CAM overlaid on the original image.
    """

    # Ensure model is in evaluation mode
    model.eval()

    # Register hook to capture the features from the last conv layer
    features = []

    def hook_function(module: LightningModule, input: Tensor, output: Tensor) -> None:
        features.append(output)

    # Register the hook to the last convolutional layer
    model.model.layer4[-1].register_forward_hook(hook_function)

    # Forward pass through the model
    _ = model(input_image.unsqueeze(0).to(model.device))

    # Get the class weights from the fully connected layer
    params = list(model.model.fc.parameters())
    weight_softmax = torch.nn.Parameter(params[0])

    # Generate CAM for the target class
    class_activation_features = torch.matmul(weight_softmax[target_class], features[0].squeeze(0).view(512, -1))

    # Reshape based on the feature map size
    feature_map_size = features[0].shape[2]  # Assuming square feature map
    cam = class_activation_features.view(feature_map_size, feature_map_size).cpu().data.numpy()
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    # Apply a heatmap color map to the CAM
    cam_heatmap = plt.get_cmap("jet")(cam)[:, :, :3]  # Slice to keep only RGB channels
    cam_heatmap = np.uint8(255 * cam_heatmap)
    cam_heatmap = Image.fromarray(cam_heatmap)

    # Resize the heatmap to match the input image size
    cam_heatmap = cam_heatmap.resize((input_image.shape[1], input_image.shape[2]), Image.BICUBIC)

    # Overlay the heatmap on the original image
    original_image = to_pil_image(input_image)
    blended_image = Image.blend(original_image, cam_heatmap, 0.25)

    return blended_image


def main(checkpoint_path: Path, sample_id: int, save_dir: Path) -> None:
    config: ClassificationConfig = get_config()

    seed_everything(seed=config.seed, workers=True)

    preprocessor = create_preprocessor(config=config.preprocessing) if config.preprocessing else None
    dm = ClassificationDataModule(config=config.data, preprocessor=preprocessor)
    dm.setup()

    test_dataset = dm.test_dataset
    if test_dataset is None:
        raise ValueError("No test dataset found. Have you run the `setup` method on you data module?")

    image, label = test_dataset[sample_id]

    model = ClassificationLightningModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        classes=DropletDrugClassificationDataset.CLASSES,
        model_config=config.model,
        loss_function_config=config.loss_function,
        optimizer_config=config.optimizer,
        metrics_config=config.metrics,
        augmentations_config=config.augmentations,
        scheduler_config=config.scheduler,
    )
    model.eval()

    class_activation_img = visualize_class_activation_map(model=model, input_image=image, target_class=label)
    class_activation_img.show()

    save_path = save_dir / f"class_activation_map_{sample_id}.png"
    save_dir.mkdir(parents=True, exist_ok=True)
    class_activation_img.save(save_path)
    _logger.info(f"Class activation map saved to {save_path}")


if __name__ == "__main__":
    experiment_dir = ARTIFACTS_DIR / "droplet-drug-classificator" / "2023-12-04_19-20-31"
    checkpoint_path = experiment_dir / "checkpoints" / "epoch=0-val_loss=0.2413.ckpt"
    save_dir = experiment_dir / "class_activation_maps"
    sample_id = 42

    main(checkpoint_path=checkpoint_path, sample_id=sample_id, save_dir=save_dir)
