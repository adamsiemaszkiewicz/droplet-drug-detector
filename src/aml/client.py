# -*- coding: utf-8 -*-
from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential

from src.common.utils.logger import get_logger

_logger = get_logger(__name__)


def get_ml_client(
    tenant_id: str,
    client_id: str,
    client_secret: str,
    subscription_id: str,
    resource_group_name: str,
    workspace_name: str,
) -> MLClient:
    """
    Creates and returns an MLClient instance for interacting with Azure Machine Learning services.

    Args:
        tenant_id (str): The Azure Active Directory tenant (directory) ID.
        client_id (str): The client (application) ID of an Azure Active Directory application.
        client_secret (str): A client secret that was generated for the App Registration in Azure AD.
        subscription_id (str): Azure subscription ID.
        resource_group_name (str): Name of the Azure resource group.
        workspace_name (str): Name of the Azure Machine Learning workspace.

    Returns:
        MLClient: An instance of MLClient for interacting with Azure Machine Learning services.
    """
    _logger.info("Initializing Azure Machine Learning client...")

    credential = ClientSecretCredential(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
    )
    _logger.info("Credentials initialized.")
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group_name,
        workspace_name=workspace_name,
    )
    _logger.info("MLClient initialized.")

    return ml_client
