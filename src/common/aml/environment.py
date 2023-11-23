# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Optional

from azure.ai.ml import MLClient
from azure.ai.ml.entities import BuildContext, Environment
from azureml.core.environment import DEFAULT_CPU_IMAGE, DEFAULT_GPU_IMAGE

from src.common.consts.directories import DOCKER_DIR
from src.common.utils.logger import get_logger

_logger = get_logger(__name__)


def build_environment(
    ml_client: MLClient,
    name: str,
    enable_gpu: bool,
    conda_dependencies_file_path: Path,
    dockerfile_path: Optional[Path] = None,
    use_dedicated_compute: bool = False,
) -> Environment:
    """
    Build or update an Azure ML Environment, optionally using a temporary compute target.

    Args:
        ml_client (MLClient): The MLClient object.
        name (str): The name of the environment.
        enable_gpu (bool): Flag to enable GPU.
        conda_dependencies_file_path (Path): The path to the conda dependencies file.
        dockerfile_path (Optional[Path]): The path to the Dockerfile.
        use_dedicated_compute (bool): Whether to use a temporary dedicated compute target for building environment.

    Returns:
        Environment: The built or updated Azure ML Environment.
    """
    _logger.info(f"Building or updating environment {name} using {conda_dependencies_file_path.as_posix()}")

    if not conda_dependencies_file_path.exists():
        raise FileNotFoundError(
            f"Conda dependencies file not found for {name}: {conda_dependencies_file_path.as_posix()}"
        )

    image = DEFAULT_GPU_IMAGE if enable_gpu else DEFAULT_CPU_IMAGE

    env = Environment(name=name, conda_file=conda_dependencies_file_path, image=image)

    if dockerfile_path is not None and dockerfile_path.exists():
        env.build = BuildContext(path=DOCKER_DIR.as_posix(), dockerfile_path=dockerfile_path.as_posix())

    if use_dedicated_compute:
        env.build_compute = "STANDARD_DS3_V2"

    created_env = ml_client.environments.create_or_update(env)

    _logger.info(f"Environment {name} is built with version: {created_env.version}")

    return created_env
