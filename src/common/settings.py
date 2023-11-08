# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from pydantic import BaseModel, BaseSettings

from src.common.consts.directories import ROOT_DIR
from src.common.utils.serialization import JsonEncoder


class AzureSettings(BaseModel):
    """The Azure settings."""

    tenant_id: Optional[str] = None
    subscription_id: Optional[str] = None
    location: Optional[str] = None
    resource_group_name: Optional[str] = None


class AzureMachineLearningSettings(BaseModel):
    """The Azure Machine Learning settings."""

    workspace_name: Optional[str] = None
    sp_client_id: Optional[str] = None
    sp_client_secret: Optional[str] = None


class DatabaseSettings(BaseModel):
    """The database settings."""

    user_name: str
    password: str
    host: str
    port: int
    name: str
    ssl_mode: str

    @property
    def uri(self) -> str:
        return f"postgresql://{self.user_name}:{self.password}@{self.host}:{self.port}/{self.name}?{self.ssl_mode}"


class BlobStorageSettings(BaseModel):
    """The Azure Blob Storage settings."""

    account_name: Optional[str] = None
    account_key: Optional[str] = None

    @property
    def connection_string(self) -> str:
        """The connection string to storage account."""
        return (
            "DefaultEndpointsProtocol=https;"
            f"AccountName={self.account_name};"
            f"AccountKey={self.account_key};"
            "EndpointSuffix=core.windows.net"
        )


class Settings(BaseSettings):
    """Serves as a container for the settings."""

    env: str
    az: AzureSettings = AzureSettings()
    aml: AzureMachineLearningSettings = AzureMachineLearningSettings()
    db: DatabaseSettings = DatabaseSettings()
    blob: BlobStorageSettings = BlobStorageSettings()

    class Config:
        env_file = ROOT_DIR / ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"

    def __str__(self) -> str:
        settings_dict = self.dict()

        for section_name in ["db", "aml", "blob"]:
            section: Dict[str, Any] = settings_dict[section_name]
            if section:
                for k, v in section.items():
                    if not v:
                        continue
                    section[k] = "**REDACTED**"

        return json.dumps(settings_dict, indent=4, cls=JsonEncoder)
