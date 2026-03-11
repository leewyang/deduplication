import time
from typing import Optional, Union, List, Callable, Dict, Any, Literal
import logging

import cudf
import ray
from ray.util import ActorPool

from gpu_shuffle import GPUShuffleActor

logging.basicConfig(
    format='%(asctime)s %(levelname)s %(name)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class GPUDataset():
    """Wrapper for a ray.data.Dataset to perform GPU-accelerated shuffling via groupby().map_groups()."""
    def __init__(self, dataset: ray.data.Dataset, nranks: int):
        self.dataset = dataset
        self.nranks = nranks
        self.actors = None
        self.t_wall_start = time.monotonic()

    def __del__(self):
        if self.dataset is not None:
            self.dataset.__del__()
        if self.actors is not None:
            self.print_stats()

    def groupby(self, key: Union[str, List[str], None], num_partitions: Optional[int] = None) -> "GPUDataset":
        self.t_wall_start = time.monotonic()
        self.key = key if isinstance(key, list) else [key]
        num_parts = num_partitions if num_partitions else self.nranks

        # create the shuffle actors
        self.actors = [
            GPUShuffleActor.remote(
                nranks=self.nranks,
                hash_parallelism=num_parts,
                group_by=self.key,
                columns=self.dataset.columns(),
                rmm_pool_size=None,  # 50% of free GPU memory
            )
            for _ in range(self.nranks)
        ]

        # setup the rapidsmpf shuffle cluster
        _, root_address = ray.get(self.actors[0].setup_root.remote())
        ray.get([actor.setup_worker.remote(root_address) for actor in self.actors])
        self.pool = ActorPool(self.actors)
        self.t_setup = time.monotonic() - self.t_wall_start
        logger.info(f"Actor pool setup complete: {self.t_setup:.2f} seconds")

        # insert dataset chunks into the actors
        t_insert_start = time.monotonic()
        batches = self.dataset.iter_batches(batch_size=1000*100, batch_format="pyarrow")
        batch_counts = list(self.pool.map(lambda actor, batch: actor.insert_batch.remote(batch), batches))
        logger.info(f"Total number of batches: {len(batch_counts)}")
        logger.info(f"Total number of items: {sum(batch_counts)}")

        # insert finished markers into the actors
        ray.get([actor.insert_finished.remote() for actor in self.actors])
        self.t_insert = time.monotonic() - t_insert_start
        logger.info(f"Actor pool insert chunks complete: {self.t_insert:.2f} seconds")
        return self

    def map_groups(
        self,
        fn: Optional[Callable[[cudf.DataFrame], cudf.DataFrame]] = None,
        fn_type: Literal["block", "group"] = "block",
        **kwargs: Dict[str, Any],
    ) -> "GPUDataset":
        """Map a GPU function over the shuffled chunks.

        Args:
            fn: The function to apply to each chunk.
            fn_type: The type of function to apply to each chunk, either "block" (entire shuffle block)
                or "group" (single group at a time), default is "block" for better GPU performance.
            **kwargs: ignored keyword arguments, for API compatibility.
        """
        # Get generators from all actors - each yields partitions as they become ready
        t_extract_start = time.monotonic()
        generators = [actor.extract_partitions.remote(fn, fn_type) for actor in self.actors]

        # Convert to iterators and seed initial refs from each
        iters = [iter(gen) for gen in generators]
        pending_refs = []
        ref_to_iter = {}

        for it in iters:
            try:
                ref = next(it)
                pending_refs.append(ref)
                ref_to_iter[ref] = it
            except StopIteration:
                pass  # Empty generator

        # Process partitions as they complete using ray.wait()
        # This allows us to receive partitions from whichever actor is ready first
        chunks = []
        total_rows = 0
        while pending_refs:
            ready_refs, pending_refs = ray.wait(pending_refs, num_returns=1)
            for ref in ready_refs:
                table = ray.get(ref)
                chunks.append(table)
                total_rows += len(table)

                # Get next ref from the same iterator
                it = ref_to_iter.pop(ref)
                try:
                    next_ref = next(it)
                    pending_refs.append(next_ref)
                    ref_to_iter[next_ref] = it
                except StopIteration:
                    pass  # Iterator exhausted

        logger.info(f"Number of chunks: {len(chunks)}")
        logger.info(f"Number of items in chunks: {total_rows}")

        # convert the list of pyarrow tables to a ray dataset
        self.dataset = ray.data.from_arrow(chunks)
        self.t_extract = time.monotonic() - t_extract_start
        logger.info(f"Actor pool read complete: {self.t_extract:.2f} seconds")

        return self

    def materialize(self) -> ray.data.Dataset:
        ds = self.dataset.materialize()
        self.t_wall_total = time.monotonic() - self.t_wall_start
        return ds

    def join(
        self,
        right_ds,
        on,
        right_on=None,
        join_type='inner',
        num_partitions=None,
        *,
        left_shuffle_chunk_rows=1_000_000,
        enable_auto_partition_adjust=True,
        enable_cpu_fallback=False,
        memory_safety_margin=0.5,
    ) -> "GPUDataset":
        """GPU-accelerated join implementation.

        Args:
            right_ds: Right dataset to join (can be Dataset or GPUDataset)
            on: Join key(s) for left dataset (string or list of strings)
            right_on: Join key(s) for right dataset (default: same as on)
            join_type: Join type - 'inner', 'left', or 'left_anti'
            num_partitions: Number of partitions for shuffle (default: nranks)
            left_shuffle_chunk_rows: Max left rows per shuffle chunk (0 = shuffle all at once).
            enable_auto_partition_adjust: Auto-increase num_partitions when partitions are too large.
            enable_cpu_fallback: On GPU OOM, retry with Ray Data CPU join.
            memory_safety_margin: Fraction of free GPU memory used as safe partition size.

        Returns:
            GPUDataset wrapping the join result

        Example:
            result = left_ds.gpu(nranks=4).join(
                right_ds, on='id', join_type='inner'
            ).materialize()
        """
        from gpu_join import GPUJoinExecutor

        # Extract underlying Dataset if right_ds is GPUDataset
        if hasattr(right_ds, 'dataset'):
            right_ds = right_ds.dataset

        # Handle tuple format (for compatibility with Ray Data API)
        if isinstance(on, tuple):
            on = list(on)
        if isinstance(right_on, tuple):
            right_on = list(right_on)

        # Normalize parameters as lists
        on_list = on if isinstance(on, list) else [on]
        right_on_list = right_on if right_on else on_list
        right_on_list = right_on_list if isinstance(right_on_list, list) else [right_on_list]

        executor = GPUJoinExecutor(
            left_ds=self.dataset,
            right_ds=right_ds,
            nranks=self.nranks,
            on=on_list,
            right_on=right_on_list,
            join_type=join_type,
            num_partitions=num_partitions or self.nranks,
            left_shuffle_chunk_rows=left_shuffle_chunk_rows,
            enable_auto_partition_adjust=enable_auto_partition_adjust,
            enable_cpu_fallback=enable_cpu_fallback,
            memory_safety_margin=memory_safety_margin,
        )

        result_ds = executor.execute()
        return GPUDataset(result_ds, nranks=self.nranks)

    def print_stats(self):
        final_stats = ray.get([actor.get_stats.remote() for actor in self.actors])

        print("\n" + "=" * 60)
        print("Timing Summary")
        print("=" * 60)
        print("\n  Driver wall-clock:")
        print(f"    {'Setup':.<30s} {self.t_setup:>8.2f}s")
        print(f"    {'Insert':.<30s} {self.t_insert:>8.2f}s")
        print(f"    {'Extract':.<30s} {self.t_extract:>8.2f}s")
        print(f"    {'Total':.<30s} {self.t_wall_total:>8.2f}s")

        # Per-actor cumulative times (these overlap due to pipelining)
        print("\n  Per-actor cumulative (pipelined - phases overlap):")
        header = (
            f"    {'Actor':>5s}  "
            f"{'Xfer Push':>10s}  {'Xfer Recv':>10s}  {'Finalize':>10s}"
        )
        print(header)
        print(f"    {'-----':>5s}  "
              f"{'----------':>10s}  {'----------':>10s}  {'----------':>10s}")
        for aid, s in enumerate(final_stats):
            print(
                f"    {aid:>5d}  "
                f"{s['time_transfer_push']:>9.2f}s  "
                f"{s['time_transfer_recv']:>9.2f}s  "
                f"{s['time_finalize']:>9.2f}s"
            )

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
    ds = self.dataset if isinstance(self, GPUDataset) else self
    return ds.materialize() if materialize else ds


def _install_extensions():
    """Monkey-patch ray.data.Dataset and GPUDataset with gpu()/cpu() methods.

    This function is called automatically on module import.
    """
    try:
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