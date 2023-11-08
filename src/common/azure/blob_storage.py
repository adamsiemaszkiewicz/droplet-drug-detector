# -*- coding: utf-8 -*-
import os
import uuid
from pathlib import Path
from typing import List, Optional

import structlog
from azure.storage.blob import BlobBlock, BlobClient, BlobProperties, BlobServiceClient, ContainerClient
from tqdm import tqdm

from src.common.consts.azure import BLOB_UPLOAD_CHUNK_SIZE

logger = structlog.get_logger()


class BlobStorageService:
    """The Azure Blob Storage Service."""

    def __init__(self, connection_string: Optional[str] = None, container_name: Optional[str] = None) -> None:
        """
        Initializes new instance of the `BlobStorageService`.

        Args:
            connection_string (str): The connection string to the Azure Blob Storage account.
            container_name (str): The name of the container to use.
        """
        self.connection_string = connection_string
        self.container_name = container_name
        self.create_container_if_not_exists()

        if not connection_string or not container_name:
            raise ValueError("Both connection_string and container_name must be provided.")

    @property
    def blob_service_client(self) -> BlobServiceClient:
        """
        Get the BlobServiceClient for the current settings.

        Returns:
            BlobServiceClient: The Azure blob service client.
        """
        return BlobServiceClient.from_connection_string(
            conn_str=self.connection_string,
            container_name=self.container_name,
        )

    @property
    def container_client(self) -> ContainerClient:
        """
        Get the container client for the current settings.

        Returns:
            ContainerClient: The Azure blob storage container client.
        """
        return self.blob_service_client.get_container_client(self.container_name)

    @property
    def all_blobs(self) -> List[BlobProperties]:
        """
        List all blob names in the container.

        Returns:
            List[BlobProperties]: A list of all blobs in the container.
        """
        return sorted(list(self.container_client.list_blobs()))

    def create_container_if_not_exists(self) -> None:
        """
        Create the container in Azure Blob Storage if it doesn't already exist.
        """
        try:
            self.container_client.create_container()
            logger.info(f"Container '{self.container_name}' created.")
        except Exception as e:
            if "ContainerAlreadyExists" in str(e):
                logger.info(f"Container '{self.container_name}' already exists.")
            else:
                raise e

    def upload_file(
        self,
        input_path: Path,
        output_path: str,
        overwrite: bool,
        chunk_size: int = BLOB_UPLOAD_CHUNK_SIZE,
    ) -> None:
        """
        Upload a file to Azure Blob Storage using manual block uploads for large files.

        Args:
            input_path (Path): Local path of the file to upload.
            output_path (str): Remote path on Azure Blob Storage where the file will be uploaded.
            overwrite (bool): Whether to overwrite existing files.
            chunk_size (int): Size of each block in bytes.
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
                    blob_client.stage_block(block_id=block_id, data=read_data)
                    block_list.append(BlobBlock(block_id=block_id))

                    pbar.update(1)

            blob_client.commit_block_list(block_list)
            logger.info(f"Successfully uploaded {input_path} to {self.container_name}, {output_path}")

        except Exception as e:
            logger.error(f"An error occurred while uploading {input_path} to {self.container_name}, {output_path}: {e}")
            raise e

    def upload_directory(self, input_dir: Path, output_dir: str, overwrite: bool = False) -> None:
        """
        Upload contents of a directory to Azure Blob Storage folder.

        Args:
            input_dir (Path): Local path of the directory containing the content to upload.
            output_dir (str): Remote path on Azure Blob Storage where the directory contents will be uploaded.
            overwrite (bool, optional): Whether to overwrite existing files. Defaults to True.
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

    def download_blob(self, blob_path: str, output_path: Path, overwrite: bool = False) -> Path:
        """
        Download a blob from Azure Blob Storage.

        Args:
            blob_path (str): Remote path of the blob on Azure Blob Storage.
            output_path (Path): Local path where the blob will be downloaded.
            overwrite (bool): Whether to overwrite existing files.

        Returns:
            Path: The local path where the blob was downloaded.
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

    def download_directory(self, blob_dir: str, output_dir: Path, overwrite: bool = False) -> List[Path]:
        """
        Download a whole directory from Azure Blob Storage.

        Args:
            blob_dir (str): Remote path on Azure Blob Storage of the directory to download.
            output_dir (Path): Local path where the directory will be downloaded.
            overwrite (bool): Whether to overwrite existing files.

        Returns:
            list[Path]: A list of local paths where the blobs were downloaded.
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
            output_path = output_dir / blob.name[len(input_dir) :]  # noqa E203
            output_path.parent.mkdir(parents=True, exist_ok=True)

            downloaded_file = self.download_blob(blob_path=blob.name, output_path=output_path, overwrite=overwrite)
            downloaded_files.append(downloaded_file)

        logger.info(f"Successfully downloaded {len(downloaded_files)} files.")

        return downloaded_files

    def delete_blob(self, blob_path: str) -> None:
        """
        Delete a remote blob from Azure Blob Storage.

        Args:
            blob_path (str): Remote path of the blob to delete.
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
        Delete a remote directory from Azure Blob Storage.

        Args:
            blob_dir (str): Remote path of the directory to delete.
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
