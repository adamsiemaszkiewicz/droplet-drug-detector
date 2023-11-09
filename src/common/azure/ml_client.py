# -*- coding: utf-8 -*-
from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential

from src.common.settings import Settings


def get_ml_client(settings: Settings) -> MLClient:
    credential = ClientSecretCredential(
        tenant_id=settings.az.tenant_id,
        client_id=settings.aml.sp_client_id,
        client_secret=settings.aml.sp_client_secret,
    )
    ml_client = MLClient(
        credential=credential,
        subscription_id=settings.az.subscription_id,
        resource_group_name=settings.az.resource_group,
        workspace_name=settings.aml.workspace_name,
    )
    return ml_client
