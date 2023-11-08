# -*- coding: utf-8 -*-
from typing import Optional

from azure.ai.ml import MLClient
from azure.ai.ml.entities import BuildContext, Environment
from azureml.core.runconfig import DEFAULT_CPU_IMAGE, DEFAULT_GPU_IMAGE

from src.common.consts.directories import DOCKER_DIR, ENVIRONMENTS_DIR
from src.common.consts.extensions import YAML
from src.common.logger import get_logger

_logger = get_logger(__name__)


def build_environment(
    ml_client: MLClient,
    name: str,
    enable_gpu: bool,
    tag: Optional[str] = None,
) -> Environment:
    """
    Build or update an Azure ML Environment.

    Args:
        ml_client (MLClient): The MLClient object.
        name (str): The name of the environment.
        enable_gpu (bool): Flag to enable GPU.
        tag (Optional[str]): The version tag for the environment.

    Returns:
        Environment: The built or updated Azure ML Environment.
    """
    _logger.info(f"Building or updating environment {name}")

    conda_dependencies_file_path = ENVIRONMENTS_DIR / f"{name}{YAML}"
    dockerfile_path = DOCKER_DIR / name / "Dockerfile"

    if not conda_dependencies_file_path.exists():
        raise FileNotFoundError(
            f"Conda dependencies file not found for {name}: {conda_dependencies_file_path.as_posix()}"
        )

    if not dockerfile_path.exists():
        _logger.warning(
            f"Dockerfile not found for {name} in {dockerfile_path.as_posix()}, "
            f"using {'GPU' if enable_gpu else 'CPU'} default image."
        )
        image = DEFAULT_GPU_IMAGE if enable_gpu else DEFAULT_CPU_IMAGE
        env = Environment(name=name, image=image, conda_file=conda_dependencies_file_path, tags={"version": tag})
    else:
        env = Environment(
            name=name,
            build=BuildContext(path=DOCKER_DIR.as_posix(), dockerfile_path=dockerfile_path),
            tags={"version": tag},
        )

    ml_client.environments.create_or_update(env)

    _logger.info(f"Current version of the '{name}' environment is: {env.version}")

    return env
