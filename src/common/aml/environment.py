# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Optional

from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute, BuildContext, Environment
from azureml.core.runconfig import DEFAULT_CPU_IMAGE, DEFAULT_GPU_IMAGE

from src.common.consts.directories import DOCKER_DIR
from src.common.utils.logger import get_logger

_logger = get_logger(__name__)


def build_environment(
    ml_client: MLClient,
    name: str,
    enable_gpu: bool,
    conda_dependencies_file_path: Path,
    use_dedicated_compute: bool,
    dockerfile_path: Optional[Path] = None,
) -> Environment:
    """
    Build or update an Azure ML Environment, optionally using a temporary compute target.

    Args:
        ml_client (MLClient): The MLClient object.
        name (str): The name of the environment.
        enable_gpu (bool): Flag to enable GPU.
        conda_dependencies_file_path (Path): The path to the conda dependencies file.
        use_dedicated_compute (bool): Whether to use a temporary compute target for building.
        dockerfile_path (Optional[Path]): The path to the Dockerfile.

    Returns:
        Environment: The built or updated Azure ML Environment.
    """

    compute_target = None
    temporary_compute_target_name = None

    if use_dedicated_compute:
        temporary_compute_target_name = f"tmp-compute-{name}"
        compute_config = AmlCompute.provisioning_configuration(vm_size="STANDARD_B2s")
        create_op = ml_client.compute.begin_create_or_update(name=temporary_compute_target_name, compute=compute_config)
        compute_target = create_op.result()

    if dockerfile_path is None or not dockerfile_path.exists():
        _logger.warning(f"Dockerfile not found for {name}, using {'GPU' if enable_gpu else 'CPU'} default image.")
        image = DEFAULT_GPU_IMAGE if enable_gpu else DEFAULT_CPU_IMAGE
        env = Environment(name=name, image=image, conda_file=conda_dependencies_file_path, compute=compute_target)
    else:
        env = Environment(
            name=name,
            build=BuildContext(path=DOCKER_DIR.as_posix(), dockerfile_path=dockerfile_path.as_posix()),
            compute=compute_target,
        )

    ml_client.environments.create_or_update(env)

    _logger.info(f"Environment '{name}' version {env.version} created/updated successfully.")

    if use_dedicated_compute:
        delete_op = ml_client.compute.begin_delete(name=temporary_compute_target_name)
        delete_op.wait()
        _logger.info(f"Temporary compute target '{temporary_compute_target_name}' deleted.")

    return env
