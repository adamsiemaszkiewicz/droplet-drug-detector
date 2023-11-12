# -*- coding: utf-8 -*-
import os
import uuid
from pathlib import Path
from typing import List

import structlog
from tqdm import tqdm

from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobBlock, BlobClient, BlobProperties, BlobServiceClient, ContainerClient
from src.common.consts.azure import BLOB_UPLOAD_CHUNK_SIZE

logger = structlog.get_logger()


class BlobStorageService:
    """
    Provides a service interface to interact with Azure Blob Storage, allowing file and directory
    operations such as upload, download, and deletion within a specified container.
    """

    def __init__(self, connection_string: str, container_name: str) -> None:
        """
        Initializes a new instance of BlobStorageService, setting up a connection and creating the
        specified container if it doesn't exist. Raises an error if both connection_string and
        container_name are not provided.

        Args:
            connection_string (str): The connection string to the Azure Blob Storage account.
            container_name (str): The name of the container to use.

        Raises:
            ValueError: If connection_string or container_name is not provided.
        """
        self.connection_string = connection_string
        self.container_name = container_name
        self.create_container_if_not_exists()

        if not connection_string or not container_name:
            raise ValueError("Both connection_string and container_name must be provided.")

    @property
    def blob_service_client(self) -> BlobServiceClient:
        """
        Retrieves the BlobServiceClient for interacting with Azure Blob Storage.

        Returns:
            BlobServiceClient: The client object for Azure Blob Service operations.
        """
        return BlobServiceClient.from_connection_string(
            conn_str=self.connection_string,
            container_name=self.container_name,
        )

    @property
    def container_client(self) -> ContainerClient:
        """
        Retrieves the ContainerClient for interacting with the specified Azure Blob Storage container.

        Returns:
            ContainerClient: The client object for container-specific operations.
        """
        return self.blob_service_client.get_container_client(self.container_name)

    @property
    def all_blobs(self) -> List[BlobProperties]:
        """
        Retrieves a sorted list of all blob properties in the container.

        Returns:
            List[BlobProperties]: A sorted list containing properties of each blob in the container.
        """
        blobs = list(self.container_client.list_blobs())
        sorted_blobs = sorted(blobs, key=lambda blob: blob.name)
        return sorted_blobs

    def create_container_if_not_exists(self) -> None:
        """
        Creates the container in Azure Blob Storage if it doesn't already exist and logs the outcome.

        Raises:
            Exception: If there's an error in creating the container, except for already existing containers.
        """
        try:
            self.container_client.create_container()
            logger.info(f"Container '{self.container_name}' created.")
        except ResourceExistsError:
            logger.info(f"Container '{self.container_name}' already exists.")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise

    def upload_file(
        self,
        input_path: Path,
        output_path: str,
        overwrite: bool,
        chunk_size: int = BLOB_UPLOAD_CHUNK_SIZE,
    ) -> None:
        """
        Uploads a file to Azure Blob Storage, supporting large file uploads through chunking.

        Args:
            input_path (Path): The local path of the file to upload.
            output_path (str): The remote path where the file will be stored in the blob container.
            overwrite (bool): Set to True to overwrite the file if it exists; False to skip upload if the file exists.
            chunk_size (int, optional): The size of each block to upload, in bytes.

        Raises:
            Exception: If there's an error during file upload.
        """
        logger.info(f"Uploading {input_path} to {self.container_name}/{output_path}")

        blob_client: BlobClient = self.container_client.get_blob_client(output_path)

        if not overwrite and blob_client.exists():
            logger.info(
                f"Skipping upload of {input_path} to {self.container_name}, {output_path} "
                "as the file already exists and overwriting is disabled"
            )
            return

        try:
            file_size = input_path.stat().st_size
            total_blocks = -(-file_size // chunk_size)  # Calculate total blocks, rounding up

            block_list = []
            with open(input_path, "rb") as src, tqdm(
                total=total_blocks, unit="block", desc="Uploading progress"
            ) as pbar:
                while True:
                    read_data = src.read(chunk_size)
                    if not read_data:
                        break  # Done reading file
                    block_id = str(uuid.uuid4())
                    blob_client.stage_block(block_id=block_id, data=read_data)  # type: ignore
                    block_list.append(BlobBlock(block_id=block_id))

                    pbar.update(1)

            blob_client.commit_block_list(block_list)
            logger.info(f"Successfully uploaded {input_path} to {self.container_name}, {output_path}")

        except Exception as e:
            logger.error(f"An error occurred while uploading {input_path} to {self.container_name}, {output_path}: {e}")
            raise e

    def upload_directory(self, input_dir: Path, output_dir: str, overwrite: bool) -> None:
        """
        Uploads the contents of a local directory to a specified directory in Azure Blob Storage recursively.

        This method will walk through the local directory's subdirectories and upload all files to the corresponding
        structure in the remote storage. The 'overwrite' parameter controls whether existing files in the destination
        are replaced with the ones being uploaded.

        Args:
            input_dir (Path): The local directory to upload from.
            output_dir (str): The remote destination directory path within the blob container.
            overwrite (bool): Whether to overwrite existing files at the destination.

        Raises:
            ValueError: If the input directory path is not a directory.
        """
        logger.info(f"Uploading directory {input_dir} to {self.container_name}, {output_dir}")

        if not input_dir.is_dir():
            raise ValueError(f"Input path {input_dir} is not a directory")

        existing_blobs = self.all_blobs
        uploaded_files_count = 0
        for root, _, files in tqdm(list(os.walk(input_dir)), desc="Uploading directories"):
            for file in tqdm(files, desc="Uploading files"):
                file_path = Path(root, file)
                output_path = file_path.as_posix().replace(input_dir.as_posix(), output_dir, 1)

                if not overwrite and output_path in existing_blobs:
                    logger.info(
                        f"Skipping upload of {file_path} to {self.container_name}, {output_path} "
                        "as the file is already uploaded and overwriting is disabled"
                    )
                    continue

                self.upload_file(input_path=file_path, output_path=output_path, overwrite=overwrite)
                uploaded_files_count += 1

        logger.info(f"Successfully uploaded {uploaded_files_count} files.")

    def download_blob(self, blob_path: str, output_path: Path, overwrite: bool) -> Path:
        """
        Downloads a single blob from Azure Blob Storage to a local file path.

        If 'overwrite' is False and the destination file already exists, the download is skipped. Otherwise,
        the blob is downloaded and saved to the local path provided, potentially overwriting the existing file.

        Args:
            blob_path (str): The remote blob path to download from.
            output_path (Path): The local file path to download the blob to.
            overwrite (bool, optional): Whether to overwrite the local file if it exists. Defaults to False.

        Returns:
            Path: The local path where the blob was downloaded to.

        Raises:
            Exception: If there's an error during the download process.
        """
        logger.info(f"Downloading {self.container_name}/{blob_path} to {output_path}")

        temp_output_path = output_path.with_suffix(".temp")

        try:
            if not overwrite and output_path.exists():
                logger.info(
                    f"Skipping download of {self.container_name}/{blob_path} to {output_path} "
                    "as the file already exists and overwriting is disabled"
                )
                return output_path

            output_path.parent.mkdir(exist_ok=True, parents=True)

            with open(temp_output_path, "wb") as download_file:
                download_file.write(self.container_client.download_blob(blob_path).readall())

            os.rename(temp_output_path, output_path)

        except Exception as e:
            logger.error(f"An error occurred while downloading {blob_path} to {output_path}: {e}")
            raise e

        finally:
            if temp_output_path.exists():
                os.remove(temp_output_path)

        logger.info("Successfully downloaded.")

        return output_path

    def download_directory(self, blob_dir: str, output_dir: Path, overwrite: bool) -> List[Path]:
        """
        Downloads the contents of a remote directory from Azure Blob Storage to a local directory.

        The directory structure of the remote storage is replicated at the local destination. Files existing at
        the destination can be optionally overwritten based on the 'overwrite' flag. This method returns a list
        of paths where files were downloaded.

        Args:
            blob_dir (str): The remote directory path to download from.
            output_dir (Path): The local directory path to download contents to.
            overwrite (bool, optional): Whether to overwrite existing files with the downloaded ones. Defaults to False.

        Returns:
            List[Path]: A list of local paths to which the blobs were downloaded.

        Raises:
            Exception: If any error occurs during the directory download process.
        """
        logger.info(f"Downloading directory {self.container_name}/{blob_dir} to {output_dir}")

        # Normalize input_dir to ensure it ends with a '/' to match full directory paths
        input_dir = blob_dir.rstrip("/") + "/"

        blobs = list(self.container_client.list_blobs(name_starts_with=input_dir))
        total_blobs = len(blobs)

        if not overwrite:
            # Get all existing files
            existing_files = set(
                os.path.join(root, file) for root, dirs, files in os.walk(str(output_dir)) for file in files
            )

            # Get blobs that don't have a corresponding local file
            blobs = [
                blob
                for blob in tqdm(blobs, desc="Checking existing files")
                if str(output_dir.absolute() / Path(blob.name[len(input_dir) :])) not in existing_files
            ]

        skipped_files_count = total_blobs - len(blobs)
        logger.info(f"Skipped {skipped_files_count} already downloaded files.")

        downloaded_files = []
        for blob in tqdm(blobs, desc="Downloading files"):
            output_path = output_dir / blob.name[len(input_dir) :]
            output_path.parent.mkdir(parents=True, exist_ok=True)

            downloaded_file = self.download_blob(blob_path=blob.name, output_path=output_path, overwrite=overwrite)
            downloaded_files.append(downloaded_file)

        logger.info(f"Successfully downloaded {len(downloaded_files)} files.")

        return downloaded_files

    def delete_blob(self, blob_path: str) -> None:
        """
        Deletes a blob from the specified Azure Blob Storage container.

        If the blob does not exist, a log entry is made, but no error is raised.

        Args:
            blob_path (str): The path of the blob to delete within the blob container.

        Notes:
            A successful deletion will be logged. Non-existence will also be logged and not treated as an error.
        """
        logger.info(f"Deleting remote blob {self.container_name}/{blob_path}")
        blob_client = self.container_client.get_blob_client(blob_path)
        if not blob_client.exists():
            logger.info(f"The blob {blob_path} does not exist.")
            return
        blob_client.delete_blob()

        logger.info("Successfully deleted.")

    def delete_directory(self, blob_dir: str) -> None:
        """
        Deletes an entire directory, including all nested blobs, from the specified Azure Blob Storage container.

        This method lists all blobs with the provided directory prefix and attempts to delete them. Failures are
        logged, and if any occur, a summary log of failed deletions is provided.

        Args:
            blob_dir (str): The remote directory path within the blob container from which all contents will be deleted.

        Notes:
            Deletion operations are logged. The method logs the total number of blobs deleted and reports individual
            deletions that failed.
        """
        logger.info(f"Deleting remote directory {self.container_name}/{blob_dir}")

        # Normalize input_dir to ensure it ends with a '/' to match full directory paths
        input_dir = blob_dir.rstrip("/") + "/"

        blobs = list(self.container_client.list_blobs(name_starts_with=input_dir))

        failed_deletions = []
        for blob in blobs:
            try:
                self.delete_blob(blob_path=blob.name)
            except Exception as e:
                logger.error(f"Failed to delete blob {blob.name}: {e}")
                failed_deletions.append(blob.name)
        if failed_deletions:
            logger.error(f"Failed to delete the following blobs: {failed_deletions}")

        logger.info(f"Successfully deleted {len(blobs)} blobs.")
