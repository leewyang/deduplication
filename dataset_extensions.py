"""
Dataset extensions for GPU/CPU mode switching via monkey-patching.

This module adds convenience methods to ray.data.Dataset and GPUDataset:
- Dataset.gpu(nranks=None): Convert to GPUDataset with optional auto-detection
- GPUDataset.cpu(materialize=False): Extract underlying Dataset
- Dataset.cpu(materialize=False): No-op for symmetry

Usage:
    import dataset_extensions  # Auto-installs methods

    # Convert to GPU
    gpu_ds = ds.gpu(nranks=4)  # Explicit
    gpu_ds = ds.gpu()          # Auto-detect

    # Use GPU operations
    result = gpu_ds.groupby(['key']).map_groups(gpu_fn).materialize()

    # Convert back to CPU
    cpu_ds = result.cpu()
"""

import logging
from typing import Optional

import ray
import ray.data

logger = logging.getLogger(__name__)


def _get_available_gpus() -> int:
    """Detect available GPUs using Ray's available_resources()."""
    try:
        if not ray.is_initialized():
            logger.debug("Ray not initialized, cannot detect GPUs")
            return 0

        resources = ray.available_resources()
        num_gpus = int(resources.get('GPU', 0))
        logger.debug(f"Detected {num_gpus} available GPUs")
        return num_gpus

    except Exception as e:
        logger.warning(f"Failed to detect GPUs: {e}")
        return 0


def _dataset_gpu(self, nranks: Optional[int] = None):
    """Convert ray.data.Dataset to GPUDataset for GPU-accelerated operations."""
    # Import here to avoid circular dependency
    from shuffle_gpu import GPUDataset

    # Prevent double-wrapping
    if isinstance(self, GPUDataset):
        return self

    # Auto-detect GPUs if not specified
    if nranks is None:
        nranks = _get_available_gpus()
        if nranks == 0:
            raise ValueError(
                "Cannot auto-detect GPUs. Either Ray is not initialized, "
                "no GPUs are available, or GPU detection failed. "
                "Please specify nranks explicitly."
            )
        logger.debug(f"Auto-detected nranks={nranks}")

    # Validate nranks
    if not isinstance(nranks, int) or nranks <= 0:
        raise ValueError(f"nranks must be a positive integer, got {nranks}")

    logger.debug(f"Converting Dataset to GPUDataset with nranks={nranks}")
    return GPUDataset(self, nranks=nranks)


def _dataset_cpu(self, materialize: bool = False):
    """No-op for ray.data.Dataset. Added for API symmetry."""
    from shuffle_gpu import GPUDataset

    ds = self.dataset if isinstance(self, GPUDataset) else self
    return ds.materialize() if materialize else ds


def _install_extensions():
    """Monkey-patch ray.data.Dataset and GPUDataset with gpu()/cpu() methods.

    This function is called automatically on module import.
    """
    try:
        # Import GPUDataset
        from shuffle_gpu import GPUDataset

        # Install Dataset.gpu()
        if not hasattr(ray.data.Dataset, 'gpu'):
            ray.data.Dataset.gpu = _dataset_gpu
            logger.debug("Installed ray.data.Dataset.gpu()")

        # Install Dataset.cpu()
        if not hasattr(ray.data.Dataset, 'cpu'):
            ray.data.Dataset.cpu = _dataset_cpu
            logger.debug("Installed ray.data.Dataset.cpu()")

        # Install GPUDataset.gpu() (for double-wrap protection)
        if not hasattr(GPUDataset, 'gpu'):
            GPUDataset.gpu = _dataset_gpu
            logger.debug("Installed GPUDataset.gpu()")

        # Install GPUDataset.cpu()
        if not hasattr(GPUDataset, 'cpu'):
            GPUDataset.cpu = _dataset_cpu
            logger.debug("Installed GPUDataset.cpu()")

    except Exception as e:
        logger.error(f"Failed to install dataset extensions: {e}")
        raise


# Auto-install on module import
try:
    _install_extensions()
except Exception as e:
    logger.error(f"Failed to install dataset extensions: {e}")
    raise
