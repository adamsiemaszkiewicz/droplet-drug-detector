# -*- coding: utf-8 -*-
import argparse
import json
from typing import Optional

from pydantic import BaseModel
from typing_extensions import Literal

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from src.azure.environment import build_environment
from src.azure.ml_client import get_ml_client
from src.common.consts.directories import ENVIRONMENTS_DIR
from src.common.consts.extensions import YAML
from src.common.settings.base import Settings
from src.common.utils.logger import get_logger

_logger = get_logger(__name__)


def _build_sample_project_environment(ml_client: MLClient, runtime_env: Literal["dev", "prod"]) -> Environment:
    """
    Build or update a specific Azure ML Environment.

    Args:
        ml_client (MLClient): The MLClient object.
        runtime_env (Literal["dev", "prod"]): The runtime environment. Can be either "dev" or "prod".

    Returns:
        Environment: The built or updated Azure ML Environment.
    """
    name = "sample-project"
    env_name = f"{runtime_env}-{name}-env"
    conda_dependencies_file_path = ENVIRONMENTS_DIR / f"{name}{YAML}"

    _logger.info(f"Building environment: {env_name}")

    return build_environment(
        ml_client=ml_client, name=env_name, enable_gpu=False, conda_dependencies_file_path=conda_dependencies_file_path
    )


class EnvironmentBuildingConfig(BaseModel):
    runtime_env: Literal["dev", "prod"]
    environment_name: str
    tag: Optional[str] = None

    def __str__(self) -> str:
        return json.dumps(self.dict(), indent=4)


def parse_args() -> EnvironmentBuildingConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime_env", type=str)
    parser.add_argument("--environment_name", type=str)
    parser.add_argument("--tag", type=str)

    args = parser.parse_args()

    cfg = EnvironmentBuildingConfig(**vars(args))

    _logger.info(f"Running with following config: {cfg}")

    return cfg


def main() -> None:
    config = parse_args()
    settings = Settings()

    ml_client = get_ml_client(
        tenant_id=settings.az.tenant_id,
        client_id=settings.aml.sp_client_id,
        client_secret=settings.aml.sp_client_secret,
        subscription_id=settings.az.subscription_id,
        resource_group_name=settings.az.resource_group,
        workspace_name=settings.aml.workspace_name,
    )

    if config.environment_name == "sample-project":
        _build_sample_project_environment(ml_client=ml_client, runtime_env=config.runtime_env)
    else:
        raise NotImplementedError(f"Unknown environment specified ({config.runtime_env}).")


if __name__ == "__main__":
    main()
