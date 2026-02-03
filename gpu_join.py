import logging
from typing import List

import ray
import ray.data

from gpu_utils import get_device_free_memory

logging.basicConfig(
    format='%(asctime)s %(levelname)s %(name)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


SUPPORTED_JOIN_TYPES = {'inner', 'left', 'left_anti'}


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
        """
        self.left_ds = left_ds
        self.right_ds = right_ds
        self.nranks = nranks
        self.on = on
        self.right_on = right_on
        self.join_type = join_type
        self.num_partitions = num_partitions

        # Validate parameters
        self._validate_join_params()

        # Materialize datasets to ensure they're computed
        logger.info("Materializing left dataset...")
        self.left_ds = self.left_ds.materialize()
        logger.info("Materializing right dataset...")
        self.right_ds = self.right_ds.materialize()

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

    def _check_partition_sizes(self):
        """Warn if partition sizes may exceed GPU memory."""
        try:
            # Get dataset sizes
            left_size = self.left_ds.size_bytes()
            right_size = self.right_ds.size_bytes()

            # Estimate partition sizes
            left_partition_size = left_size / self.num_partitions
            right_partition_size = right_size / self.num_partitions

            # Get GPU memory
            gpu_memory = get_device_free_memory()
            if gpu_memory is None:
                logger.warning("Could not determine GPU memory, skipping partition size check")
                return

            # Warn if partitions are large (use 50% of GPU memory as threshold)
            threshold = gpu_memory * 0.5
            max_partition_size = max(left_partition_size, right_partition_size)

            if max_partition_size > threshold:
                logger.warning(
                    f"Partition size ({max_partition_size / 1e9:.2f} GB) may exceed "
                    f"available GPU memory ({gpu_memory / 1e9:.2f} GB). "
                    f"Consider increasing num_partitions from {self.num_partitions}."
                )
            else:
                logger.info(
                    f"Partition size check passed: max partition size "
                    f"{max_partition_size / 1e9:.2f} GB, GPU memory: "
                    f"{gpu_memory / 1e9:.2f} GB"
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
        # Import here to avoid circular dependency
        from shuffle_gpu import GPUJoinActor

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

        # PHASE 2: Shuffle left dataset
        logger.info("=" * 60)
        logger.info("PHASE 2: Shuffling left dataset...")
        logger.info("=" * 60)
        self._shuffle_dataset_into_actors(actors, self.left_ds, self.on, is_right=False)
        logger.info("Completing left dataset shuffle...")
        ray.get([actor.left_insert_finished.remote() for actor in actors])
        logger.info("Phase 2 complete: Left dataset shuffled")

        # PHASE 3: Execute join on each actor
        logger.info("=" * 60)
        logger.info("PHASE 3: Executing join on GPU actors...")
        logger.info("=" * 60)
        result_tables = ray.get([actor.execute_join.remote() for actor in actors])

        # Filter out empty results
        result_tables = [t for t in result_tables if len(t) > 0]

        if len(result_tables) == 0:
            logger.info("No matching rows found, returning empty dataset")
            return ray.data.from_items([])

        logger.info(f"Join complete: {len(result_tables)} result partitions")
        logger.info(f"Total rows: {sum(len(t) for t in result_tables)}")

        # Convert results to Dataset
        result_ds = ray.data.from_arrow(result_tables)
        return result_ds

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
