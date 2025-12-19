import os
import glob
from typing import List
import pyarrow.fs as pafs
import logging

logger = logging.getLogger(__name__)


def _is_gcs_path(path: str) -> bool:
    """Check if a path is a GCS path."""
    return path.startswith("gs://")


def check_path_exists(path: str) -> bool:
    """Check if a path exists (supports both GCS and local paths)."""
    if _is_gcs_path(path):
        return check_gcs_path_exists(path)
    else:
        return check_local_path_exists(path)


def check_local_path_exists(path: str) -> bool:
    """Check if a local path exists."""
    return os.path.exists(path)


def check_gcs_path_exists(path: str) -> bool:
    """Check if a GCS path exists."""
    gcs_fs = pafs.GcsFileSystem()

    # Remove gs:// prefix if present
    if path.startswith("gs://"):
        path = path[5:]

    # Remove trailing slash
    path = path.rstrip("/")

    logger.info(f"Listing parquet files in gs://{path}")

    # Use FileSelector with recursive=True
    selector = pafs.FileSelector(path, recursive=True)
    file_infos = gcs_fs.get_file_info(selector)
    return len(file_infos) > 0


def list_parquet_files(path: str) -> List[str]:
    """
    List all parquet files in a directory recursively (supports both GCS and local paths).

    Args:
        path: Path to directory (GCS path like gs://bucket/path/ or local path)

    Returns:
        List of full paths to parquet files
    """
    if _is_gcs_path(path):
        return list_gcs_parquet_files(path)
    else:
        return list_local_parquet_files(path)


def list_local_parquet_files(path: str) -> List[str]:
    """
    List all parquet files in a local directory recursively.

    Args:
        path: Local path to directory

    Returns:
        List of full local paths to parquet files
    """
    path = path.rstrip("/")
    logger.info(f"Listing parquet files in {path}")

    parquet_files = []

    if os.path.isfile(path):
        # Single file
        if path.endswith('.parquet'):
            parquet_files.append(path)
    elif os.path.isdir(path):
        # Directory - search recursively
        pattern = os.path.join(path, "**", "*.parquet")
        parquet_files = glob.glob(pattern, recursive=True)
    else:
        # Could be a glob pattern
        parquet_files = [f for f in glob.glob(path, recursive=True) if f.endswith('.parquet')]

    logger.info(f"Found {len(parquet_files)} parquet files")
    return parquet_files


def list_gcs_parquet_files(path: str) -> List[str]:
    """
    List all parquet files in a GCS directory recursively using PyArrow.

    Args:
        path: GCS path (e.g., gs://bucket/path/)

    Returns:
        List of full GCS paths to parquet files
    """
    # Create GCS filesystem
    gcs_fs = pafs.GcsFileSystem()

    # Remove gs:// prefix if present
    if path.startswith("gs://"):
        path = path[5:]

    # Remove trailing slash
    path = path.rstrip("/")

    logger.info(f"Listing parquet files in gs://{path}")

    # Use FileSelector with recursive=True
    selector = pafs.FileSelector(path, recursive=True)
    file_infos = gcs_fs.get_file_info(selector)

    # Filter for parquet files
    parquet_files = []
    for file_info in file_infos:
        if file_info.type == pafs.FileType.File and file_info.path.endswith('.parquet'):
            parquet_files.append(f"gs://{file_info.path}")

    logger.info(f"Found {len(parquet_files)} parquet files")

    return parquet_files
