# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Optional

from azure.ai.ml import MLClient
from azure.ai.ml.entities import BuildContext, Environment
from azureml.core.runconfig import DEFAULT_CPU_IMAGE, DEFAULT_GPU_IMAGE

from src.common.consts.directories import DOCKER_DIR
from src.common.utils.logger import get_logger

_logger = get_logger(__name__)


def build_environment(
    ml_client: MLClient,
    name: str,
    enable_gpu: bool,
    conda_dependencies_file_path: Path,
    dockerfile_path: Optional[Path] = None,
) -> Environment:
    """
    Build or update an Azure ML Environment.

    Args:
        ml_client (MLClient): The MLClient object.
        name (str): The name of the environment.
        enable_gpu (bool): Flag to enable GPU.
        conda_dependencies_file_path (Path): The path to the conda dependencies file.
        dockerfile_path (Optional[Path]): The path to the Dockerfile.

    Returns:
        Environment: The built or updated Azure ML Environment.
    """
    _logger.info(f"Building or updating environment {name} using {conda_dependencies_file_path.as_posix()}")

    if not conda_dependencies_file_path.exists():
        raise FileNotFoundError(
            f"Conda dependencies file not found for {name}: {conda_dependencies_file_path.as_posix()}"
        )

    if dockerfile_path is None or not dockerfile_path.exists():
        _logger.warning(f"Dockerfile not found for {name}, using {'GPU' if enable_gpu else 'CPU'} default image.")
        image = DEFAULT_GPU_IMAGE if enable_gpu else DEFAULT_CPU_IMAGE
        env = Environment(name=name, image=image, conda_file=conda_dependencies_file_path)
    else:
        env = Environment(
            name=name,
            build=BuildContext(dockerfile_path=dockerfile_path.as_posix(), path=DOCKER_DIR.as_posix()),
        )

    ml_client.environments.create_or_update(env)

    _logger.info(f"Current version of the '{name}' environment is: {env.version}")

    return env
