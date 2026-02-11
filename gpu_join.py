from typing import List, Tuple
import logging

import cudf
import ray
import pyarrow as pa
from ray.exceptions import RayTaskError

from gpu_utils import get_device_free_memory
from rapidsmpf_shuffler import BulkRapidsMPFShuffler
from rapidsmpf.utils.cudf import pylibcudf_to_cudf_dataframe

logging.basicConfig(
    format='%(asctime)s %(levelname)s %(name)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


SUPPORTED_JOIN_TYPES = {'inner', 'left', 'left_anti'}


@ray.remote(num_gpus=1)
class GPUJoinActor(BulkRapidsMPFShuffler):
    """GPU actor for sort-merge join operations using two-phase shuffle.

    Phase 1: Shuffle right dataset by right_keys, extract and store partitions
    Phase 2: Shuffle left dataset by left_keys, join with stored right partitions

    This ensures both datasets are properly hash-partitioned by their join keys,
    avoiding memory issues with large right datasets.
    """

    def __init__(
        self,
        nranks: int,
        hash_parallelism: int,
        left_keys: List[str],
        right_keys: List[str],
        left_columns: List[str],
        right_columns: List[str],
        join_type: str,
    ):
        """Initialize GPU join actor.

        Args:
            nranks: Total number of GPU actors
            hash_parallelism: Number of partitions for hash-based shuffle
            left_keys: Join keys for left dataset
            right_keys: Join keys for right dataset
            left_columns: All columns in left dataset
            right_columns: All columns in right dataset
            join_type: Type of join ('inner', 'left', 'left_anti')
        """
        # Initialize with RIGHT keys for Phase 1 shuffle
        super().__init__(nranks=nranks, total_nparts=hash_parallelism, shuffle_on=right_keys)

        self.left_keys = left_keys
        self.right_keys = right_keys
        self.left_columns = left_columns
        self.right_columns = right_columns
        self.join_type = join_type
        self.hash_parallelism = hash_parallelism

        # Storage for shuffled right dataset partitions (cuDF DataFrames)
        self.stored_right_partitions = []
        self.phase = 'right'  # Track current phase: 'right' or 'left'

        logger.info(f"GPU join actor initialized in Phase 1 (right shuffle) mode")

    def insert_right_batch(self, batch: pa.Table) -> int:
        """Phase 1: Insert right batch into RAPIDS MPF shuffler."""
        if self.phase != 'right':
            raise RuntimeError(f"Cannot insert right batch in phase '{self.phase}'")
        df = cudf.DataFrame.from_arrow(batch)
        self.insert_chunk(table=df, column_names=self.right_columns)
        return len(batch)

    def right_insert_finished(self):
        """Phase 1: Complete right shuffle, extract and store partitions."""
        if self.phase != 'right':
            raise RuntimeError(f"Cannot finish right insert in phase '{self.phase}'")

        rank = self.rank()
        logger.info(f"Rank {rank}: Finishing right dataset shuffle...")

        # Mark insertion as finished
        self.insert_finished()

        # Extract and store shuffled right partitions
        logger.info(f"Rank {rank}: Extracting right partitions...")
        right_partitions = []
        for partition_id, partition in self.extract():
            cdf = pylibcudf_to_cudf_dataframe(partition, self.right_columns)
            right_partitions.append(cdf)
        self.stored_right_df = cudf.concat(right_partitions)

        # Reset for Phase 2 (left dataset shuffle)
        self._reset_for_left_shuffle()

    def _reset_for_left_shuffle(self):
        """Create new shuffler for Phase 2 (left dataset)."""
        rank = self.rank()
        logger.info(f"Rank {rank}: Resetting shuffler for Phase 2 (left shuffle)...")

        # Update shuffle keys to left_keys
        self.shuffle_on = self.left_keys

        # Create new shuffler instance (reuses existing buffer resources and comm)
        self.shuffler = self.create_shuffler(
            0,
            total_num_partitions=self.hash_parallelism,
            buffer_resource=self.br,
            statistics=self.stats,
        )

        self.phase = 'left'
        logger.info(f"Rank {rank}: Phase 2 initialized (left shuffle mode)")

    def insert_left_batch(self, batch: pa.Table) -> int:
        """Phase 2: Insert left batch into RAPIDS MPF shuffler."""
        if self.phase != 'left':
            raise RuntimeError(f"Cannot insert left batch in phase '{self.phase}'. Must finish right shuffle first.")
        df = cudf.DataFrame.from_arrow(batch)
        self.insert_chunk(table=df, column_names=self.left_columns)
        return len(batch)

    def left_insert_finished(self):
        """Phase 2: Mark left dataset insert as finished."""
        if self.phase != 'left':
            raise RuntimeError(f"Cannot finish left insert in phase '{self.phase}'")
        self.insert_finished()
        logger.info(f"Rank {self.rank()}: Left dataset insert finished")

    def start_left_chunk(self):
        """Reset left shuffler for the next chunk of left data (chunked shuffle).

        Call after execute_join() when processing left in multiple chunks so the actor
        can accept another round of insert_left_batch / left_insert_finished / execute_join.
        """
        self._reset_for_left_shuffle()

    def execute_join(
        self,
    ) -> List[pa.Table]:
        """Phase 3: Extract left partitions in chunks and join with stored right partitions.

        Returns:
            List of PyArrow tables with join results, one per partition.
        """
        if self.phase != 'left':
            raise RuntimeError(f"Cannot execute join in phase '{self.phase}'. Must complete both shuffles first.")

        results: List[pa.Table] = []
        for idx, (partition_id, left_partition) in enumerate(self.extract()):
            left_cdf = pylibcudf_to_cudf_dataframe(left_partition, self.left_columns)
            if len(left_cdf) == 0:
                continue
            result = self._merge_join(left_cdf, self.stored_right_df)
            results.append(result.to_arrow(preserve_index=False))

        return results

    def _merge_join(
        self,
        left_cdf: cudf.DataFrame,
        right_cdf: cudf.DataFrame,
    ) -> cudf.DataFrame:
        """Perform sort-merge join on sorted DataFrames.

        For left_anti with large left side, processes left in chunks to stay within GPU memory.

        Args:
            left_cdf: Sorted left DataFrame
            right_cdf: Sorted right DataFrame

        Returns:
            Joined DataFrame
        """
        join_mapping = {
            'inner': 'inner',
            'left': 'left',
            'left_anti': 'leftanti',
        }
        if self.join_type in join_mapping:
            how = join_mapping[self.join_type]
        else:
            raise ValueError(f"Unsupported join type: {self.join_type}")

        return left_cdf.merge(
            right_cdf,
            left_on=self.left_keys,
            right_on=self.right_keys,
            how=how,
        )


class GPUJoinExecutor:
    """Orchestrates GPU-accelerated join execution using GPUShuffleActor pool."""

    def __init__(
        self,
        left_ds: ray.data.Dataset,
        right_ds: ray.data.Dataset,
        nranks: int,
        on: List[str],
        right_on: List[str],
        join_type: str,
        num_partitions: int,
        *,
        left_shuffle_chunk_rows: int = 1_000_000,
        enable_auto_partition_adjust: bool = True,
        enable_cpu_fallback: bool = False,
        memory_safety_margin: float = 0.5,
    ):
        """Initialize GPU join executor.

        Args:
            left_ds: Left dataset to join
            right_ds: Right dataset to join
            nranks: Number of GPU actors to use
            on: Join keys for left dataset
            right_on: Join keys for right dataset
            join_type: Join type ('inner', 'left', 'left_anti')
            num_partitions: Number of partitions for shuffle (default: nranks)
            left_shuffle_chunk_rows: Max left rows per shuffle chunk (0 = shuffle all left at once).
            enable_auto_partition_adjust: If True, increase num_partitions when estimated size exceeds GPU memory.
            enable_cpu_fallback: If True, on GPU OOM retry with Ray Data CPU join.
            memory_safety_margin: Use this fraction of free GPU memory as safe partition size (0.5 = 50%).
        """
        self.left_ds = left_ds
        self.right_ds = right_ds
        self.nranks = nranks
        self.on = on
        self.right_on = right_on
        self.join_type = join_type
        self.num_partitions = num_partitions
        self.left_shuffle_chunk_rows = left_shuffle_chunk_rows
        self.enable_auto_partition_adjust = enable_auto_partition_adjust
        self.enable_cpu_fallback = enable_cpu_fallback
        self.memory_safety_margin = memory_safety_margin

        # Validate parameters
        self._validate_join_params()

        # Materialize datasets to ensure they're computed
        logger.info("Materializing left dataset...")
        self.left_ds = self.left_ds.materialize()
        logger.info("Materializing right dataset...")
        self.right_ds = self.right_ds.materialize()

        # Optionally increase partitions so each fits in GPU memory
        self._maybe_adjust_partitions()
        # Estimate partition sizes and warn if needed
        self._check_partition_sizes()

    def _validate_join_params(self):
        """Validate join keys exist in both datasets and join type is supported."""
        # Check join type
        if self.join_type not in SUPPORTED_JOIN_TYPES:
            raise ValueError(
                f"Unsupported join type: '{self.join_type}'. "
                f"Supported types: {SUPPORTED_JOIN_TYPES}"
            )

        # Get column names (handle empty datasets)
        left_cols = self.left_ds.columns()
        right_cols = self.right_ds.columns()

        if left_cols is None:
            left_cols = []
        if right_cols is None:
            right_cols = []

        # Skip validation for empty datasets (will be handled in execute())
        if len(left_cols) == 0 or len(right_cols) == 0:
            logger.info("Skipping key validation for empty dataset")
            return

        # Validate left keys
        for key in self.on:
            if key not in left_cols:
                raise ValueError(
                    f"Join key '{key}' not found in left dataset. "
                    f"Available columns: {left_cols}"
                )

        # Validate right keys
        for key in self.right_on:
            if key not in right_cols:
                raise ValueError(
                    f"Join key '{key}' not found in right dataset. "
                    f"Available columns: {right_cols}"
                )

        # Validate key counts match
        if len(self.on) != len(self.right_on):
            raise ValueError(
                f"Number of left keys ({len(self.on)}) must match "
                f"number of right keys ({len(self.right_on)})"
            )

        logger.info(f"Join validation passed: {self.on} = {self.right_on}, join_type={self.join_type}")

    def _maybe_adjust_partitions(self):
        """Increase num_partitions if estimated partition size exceeds safe GPU memory."""
        if not self.enable_auto_partition_adjust:
            return
        try:
            gpu_memory = get_device_free_memory()
            if gpu_memory is None:
                return
            threshold = gpu_memory * self.memory_safety_margin
            left_size = self.left_ds.size_bytes()
            right_size = self.right_ds.size_bytes()
            # Per-actor: each GPU gets ~1/nranks of total data, split into num_partitions/nranks partitions
            # So partition size per actor ~ (left_size + right_size) / num_partitions
            total_size = left_size + right_size
            partition_size = total_size / self.num_partitions
            max_partitions = 100_000
            while partition_size > threshold and self.num_partitions < max_partitions:
                old = self.num_partitions
                self.num_partitions = min(self.num_partitions * 2, max_partitions)
                partition_size = total_size / self.num_partitions
                if self.num_partitions == old:
                    break
                logger.warning(
                    f"Auto-adjusting num_partitions: {old} -> {self.num_partitions} "
                    f"(partition size {partition_size / 1e9:.2f} GB > threshold {threshold / 1e9:.2f} GB)"
                )
        except Exception as e:
            logger.warning(f"Failed to auto-adjust partitions: {e}")

    def _check_partition_sizes(self):
        """Warn if partition sizes may exceed GPU memory."""
        try:
            left_size = self.left_ds.size_bytes()
            right_size = self.right_ds.size_bytes()
            left_partition_size = left_size / self.num_partitions
            right_partition_size = right_size / self.num_partitions
            gpu_memory = get_device_free_memory()
            if gpu_memory is None:
                logger.warning("Could not determine GPU memory, skipping partition size check")
                return
            threshold = gpu_memory * self.memory_safety_margin
            max_partition_size = max(left_partition_size, right_partition_size)
            if max_partition_size > threshold:
                logger.warning(
                    f"Partition size ({max_partition_size / 1e9:.2f} GB) may exceed "
                    f"safe threshold ({threshold / 1e9:.2f} GB). "
                    f"num_partitions={self.num_partitions}. Enable enable_auto_partition_adjust or increase num_partitions."
                )
            else:
                logger.info(
                    f"Partition size check passed: max {max_partition_size / 1e9:.2f} GB, "
                    f"threshold {threshold / 1e9:.2f} GB"
                )
        except Exception as e:
            logger.warning(f"Failed to check partition sizes: {e}")

    def execute(self) -> ray.data.Dataset:
        """Execute the two-phase sort-merge join algorithm.

        Phase 1: Shuffle right dataset by right_keys, store partitions
        Phase 2: Shuffle left dataset by left_keys
        Phase 3: Join left partitions with stored right partitions

        Returns:
            ray.data.Dataset containing the join results
        """
        logger.info(f"Starting two-phase GPU join: {self.join_type} join on {self.on} = {self.right_on}")
        logger.info(f"Using {self.nranks} GPUs with {self.num_partitions} partitions")

        # Handle empty datasets
        left_count = self.left_ds.count()
        right_count = self.right_ds.count()

        if left_count == 0:
            logger.info("Left dataset is empty, returning empty dataset")
            return ray.data.from_items([])

        if right_count == 0:
            if self.join_type in ('inner',):
                logger.info("Right dataset is empty, returning empty dataset for inner join")
                return ray.data.from_items([])
            elif self.join_type == 'left':
                logger.info("Right dataset is empty, returning left dataset for left join")
                return self.left_ds
            elif self.join_type == 'left_anti':
                logger.info("Right dataset is empty, returning left dataset for left_anti join")
                return self.left_ds

        # Create GPU join actors
        logger.info("Creating GPU join actors...")
        actors = [
            GPUJoinActor.remote(
                nranks=self.nranks,
                hash_parallelism=self.num_partitions,
                left_keys=self.on,
                right_keys=self.right_on,
                left_columns=self.left_ds.columns(),
                right_columns=self.right_ds.columns(),
                join_type=self.join_type,
            )
            for _ in range(self.nranks)
        ]

        # Setup RAPIDS MPF cluster
        logger.info("Setting up RAPIDS MPF cluster...")
        _, root_address = ray.get(actors[0].setup_root.remote())
        ray.get([actor.setup_worker.remote(root_address) for actor in actors])

        # PHASE 1: Shuffle right dataset
        logger.info("=" * 60)
        logger.info("PHASE 1: Shuffling right dataset...")
        logger.info("=" * 60)
        self._shuffle_dataset_into_actors(actors, self.right_ds, self.right_on, is_right=True)
        logger.info("Completing right dataset shuffle and storing partitions...")
        ray.get([actor.right_insert_finished.remote() for actor in actors])
        logger.info("Phase 1 complete: Right dataset shuffled and stored")

        # PHASE 2 + 3: Shuffle left (in chunks if requested) and execute join per chunk
        result_tables = []
        total_rows = 0

        try:
            if self.left_shuffle_chunk_rows <= 0:
                # Single shot: shuffle all left, then one execute_join
                logger.info("=" * 60)
                logger.info("PHASE 2: Shuffling left dataset (full)...")
                logger.info("=" * 60)
                self._shuffle_dataset_into_actors(actors, self.left_ds, self.on, is_right=False)
                logger.info("Completing left dataset shuffle...")
                ray.get([actor.left_insert_finished.remote() for actor in actors])
                logger.info("Phase 2 complete: Left dataset shuffled")
                logger.info("=" * 60)
                logger.info("PHASE 3: Executing join on GPU actors...")
                logger.info("=" * 60)
                chunk_tables, chunk_rows = self._collect_join_results(actors)
                result_tables.extend(chunk_tables)
                total_rows += chunk_rows
            else:
                # Chunked: shuffle left in chunks of left_shuffle_chunk_rows, join each chunk
                target_chunk_rows = self.left_shuffle_chunk_rows
                chunk_index = 0

                for batch in self.left_ds.iter_batches(batch_size=target_chunk_rows, batch_format="pyarrow"):
                    # Shuffle this chunk and run join
                    chunk_ds = ray.data.from_arrow(batch)
                    logger.info("=" * 60)
                    logger.info(
                        f"PHASE 2+3 (chunk {chunk_index}): Shuffling left chunk ({len(batch)} rows)..."
                    )
                    logger.info("=" * 60)
                    ray.get([actor.start_left_chunk.remote() for actor in actors])
                    self._shuffle_dataset_into_actors(actors, chunk_ds, self.on, is_right=False)
                    ray.get([actor.left_insert_finished.remote() for actor in actors])
                    ct, cr = self._collect_join_results(actors)
                    result_tables.extend(ct)
                    total_rows += cr
                    logger.info(f"Chunk {chunk_index} complete: {cr} result rows (cumulative {total_rows})")
                    chunk_index += 1

            if len(result_tables) == 0:
                logger.info("No matching rows found, returning empty dataset")
                return ray.data.from_items([])

            logger.info(f"Join complete: {len(result_tables)} result partitions, {total_rows} total rows")
            return ray.data.from_arrow(result_tables)

        except (MemoryError, RayTaskError) as e:
            cause = getattr(e, "cause", None) or getattr(e, "__cause__", None)
            is_oom = isinstance(e, MemoryError) or isinstance(cause, MemoryError)
            if is_oom and self.enable_cpu_fallback:
                logger.warning(
                    "GPU join failed with out-of-memory, falling back to Ray Data CPU join: %s",
                    e,
                )
                return self._execute_cpu_join()
            raise

    def _collect_join_results(self, actors: List) -> Tuple[List[pa.Table], int]:
        """Run Phase 3 (execute_join) on actors and collect all result tables.

        Returns:
            (result_tables, total_rows)
        """
        # Each actor returns List[pa.Table]; flatten into one list of tables
        per_actor_results = ray.get([actor.execute_join.remote() for actor in actors])
        result_tables = [t for tables in per_actor_results for t in tables]
        return result_tables, sum(len(table) for table in result_tables)

    def _execute_cpu_join(self) -> ray.data.Dataset:
        """Run join on CPU using Ray Data native implementation."""
        return self.left_ds.join(
            self.right_ds,
            on=tuple(self.on),
            right_on=tuple(self.right_on),
            join_type=self.join_type,
            num_partitions=self.num_partitions,
        )

    def _shuffle_dataset_into_actors(
        self,
        actors: List,
        ds: ray.data.Dataset,
        keys: List[str],
        is_right: bool,
    ):
        """Shuffle dataset into GPU actors using RAPIDS MPF.

        Args:
            actors: List of GPUJoinActor instances
            ds: Dataset to shuffle
            keys: Keys to shuffle on
            is_right: True if this is the right dataset, False for left
        """
        from ray.util import ActorPool

        pool = ActorPool(actors)

        # Insert batches into actors
        batches = ds.iter_batches(batch_size=1000 * 100, batch_format="pyarrow")
        if is_right:
            batch_counts = list(pool.map(lambda actor, batch: actor.insert_right_batch.remote(batch), batches))
        else:
            batch_counts = list(pool.map(lambda actor, batch: actor.insert_left_batch.remote(batch), batches))

        logger.info(f"Shuffled {sum(batch_counts)} rows for {'right' if is_right else 'left'} dataset")
