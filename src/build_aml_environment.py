# -*- coding: utf-8 -*-
import argparse
import json
from typing import Optional

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from pydantic import BaseModel
from typing_extensions import Literal

from src.common.azure.environment import build_environment
from src.common.azure.ml_client import get_ml_client
from src.common.consts.directories import ENVIRONMENTS_DIR
from src.common.consts.extensions import YAML
from src.common.logger import get_logger
from src.common.settings import Settings

_logger = get_logger(__name__)


def _build_sample_environment(
    ml_client: MLClient, runtime_env: Literal["dev", "prod"], tag: Optional[str] = None
) -> Environment:
    """
    Build or update a specific Azure ML Environment.

    Args:
        ml_client (MLClient): The MLClient object.
        runtime_env (Literal["dev", "prod"]): The runtime environment. Can be either "dev" or "prod".
        tag (str): The version tag for the environment.

    Returns:
        Environment: The built or updated Azure ML Environment.
    """
    name = "sample"
    env_name = f"{runtime_env}-{name}-env"
    conda_dependencies_file_path = ENVIRONMENTS_DIR / f"{name}{YAML}"

    _logger.info(f"Building environment: {env_name}")

    return build_environment(
        ml_client=ml_client,
        name=name,
        enable_gpu=False,
        conda_dependencies_file_path=conda_dependencies_file_path,
        tag=tag,
    )


class EnvironmentBuildingConfig(BaseModel):
    runtime_env: Literal["dev", "prod"]
    sample: bool
    tag: Optional[str] = None

    def __str__(self) -> str:
        return json.dumps(self.dict(), indent=4)


def parse_args() -> EnvironmentBuildingConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime_env", type=str)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--tag", type=str)

    args = parser.parse_args()

    cfg = EnvironmentBuildingConfig(**vars(args))

    _logger.info(f"Running with following config: {cfg}")

    return cfg


def main() -> None:
    config = parse_args()
    settings = Settings()

    ml_client = get_ml_client(settings=settings)

    if config.sample:
        _build_sample_environment(ml_client=ml_client, runtime_env=config.runtime_env, tag=config.tag)
    else:
        raise NotImplementedError(f"Unknown environment specified ({config.runtime_env}).")


if __name__ == "__main__":
    main()
