from typing import Any, Callable, Dict, List, Literal, Optional, Union

import os
import logging
import time

import cudf
import cupy as cp
import fire
import numpy as np
import pyarrow as pa
import ray
from ray.util import ActorPool

from rapidsmpf.utils.cudf import pylibcudf_to_cudf_dataframe
from rapidsmpf_shuffler import BulkRapidsMPFShuffler

logging.basicConfig(
    format='%(asctime)s %(levelname)s %(name)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


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
        for partition_id, partition in self.extract():
            cdf = pylibcudf_to_cudf_dataframe(partition, self.right_columns)
            self.stored_right_partitions.append(cdf)
            logger.info(f"Rank {rank}: Stored right partition {partition_id} with {len(cdf)} rows")

        total_right_rows = sum(len(cdf) for cdf in self.stored_right_partitions)
        logger.info(f"Rank {rank}: Phase 1 complete - stored {len(self.stored_right_partitions)} partitions, {total_right_rows} total rows")

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

    def execute_join(self) -> pa.Table:
        """Phase 3: Extract left partitions and join with stored right partitions.

        Returns:
            PyArrow table with join results
        """
        if self.phase != 'left':
            raise RuntimeError(f"Cannot execute join in phase '{self.phase}'. Must complete both shuffles first.")

        rank = self.rank()
        logger.info(f"Rank {rank}: Phase 3 - Executing join...")

        # Prepare right dataset (already shuffled and stored)
        if len(self.stored_right_partitions) > 0:
            logger.info(f"Rank {rank}: Concatenating {len(self.stored_right_partitions)} stored right partitions...")
            right_cdf = cudf.concat(self.stored_right_partitions)
            logger.info(f"Rank {rank}: Right partition size: {len(right_cdf)} rows")

            # Sort right by join keys
            logger.info(f"Rank {rank}: Sorting right partition by {self.right_keys}...")
            right_cdf = right_cdf.sort_values(by=self.right_keys)
        else:
            logger.info(f"Rank {rank}: No right partitions stored")
            right_cdf = None

        # Extract and process left partitions
        result_parts = []
        logger.info(f"Rank {rank}: Extracting and joining left partitions...")

        for partition_id, left_partition in self.extract():
            # Convert to cuDF
            left_cdf = pylibcudf_to_cudf_dataframe(left_partition, self.left_columns)
            logger.info(f"Rank {rank}: Processing left partition {partition_id} with {len(left_cdf)} rows")

            if len(left_cdf) == 0:
                continue

            # Sort left by join keys
            left_cdf = left_cdf.sort_values(by=self.left_keys)

            # Perform join
            if right_cdf is not None and len(right_cdf) > 0:
                result = self._merge_join(left_cdf, right_cdf)
            else:
                # Handle empty right dataset based on join type
                if self.join_type in ('left', 'left_anti'):
                    result = left_cdf
                else:  # inner join
                    result = cudf.DataFrame({col: [] for col in self.left_columns})

            if len(result) > 0:
                result_parts.append(result)

        # Combine all result parts
        if len(result_parts) == 0:
            logger.info(f"Rank {rank}: No matching rows found, returning empty result")
            return pa.table({col: [] for col in self.left_columns})

        logger.info(f"Rank {rank}: Concatenating {len(result_parts)} result parts...")
        final_result = cudf.concat(result_parts)
        logger.info(f"Rank {rank}: Join complete, final result size: {len(final_result)} rows")

        return final_result.to_arrow(preserve_index=False)

    def _merge_join(self, left_cdf: cudf.DataFrame, right_cdf: cudf.DataFrame) -> cudf.DataFrame:
        """Perform sort-merge join on sorted DataFrames.

        Args:
            left_cdf: Sorted left DataFrame
            right_cdf: Sorted right DataFrame

        Returns:
            Joined DataFrame
        """
        if self.join_type in ['inner', 'left']:
            result = left_cdf.merge(
                right_cdf,
                left_on=self.left_keys,
                right_on=self.right_keys,
                how=self.join_type,
            )
        elif self.join_type == 'left_anti':
            # cuDF doesn't support indicator parameter, so implement anti join manually
            # Add a marker column to right dataset
            right_cdf_marked = right_cdf.copy()
            right_cdf_marked['__right_marker__'] = 1

            # Perform left join
            temp = left_cdf.merge(
                right_cdf_marked[[*self.right_keys, '__right_marker__']],
                left_on=self.left_keys,
                right_on=self.right_keys,
                how='left',
            )

            # Keep only rows where marker is null (no match in right)
            result = temp[temp['__right_marker__'].isna()][self.left_columns]
        else:
            raise ValueError(f"Unsupported join type: {self.join_type}")

        return result


@ray.remote(num_gpus=1)
class GPUShuffleActor(BulkRapidsMPFShuffler):
    def __init__(
        self,
        nranks: int,
        hash_parallelism: int,
        group_by: list[str],
        columns: list[str],
        **kwargs: Any,
    ):
        super().__init__(nranks=nranks, total_nparts=hash_parallelism, shuffle_on=group_by, **kwargs)
        self.columns = columns
        self.group_by = group_by
        logger.info(f"Rank {self.rank} setup complete")

    def insert_batch(self, batch: pa.Table) -> int:
        df = cudf.DataFrame.from_arrow(batch)
        # self.columns = list(df.columns)
        self.insert_chunk(table=df, column_names=self.columns)
        return len(batch)

    def extract_partitions(
        self,
        map_groups_fn: Optional[Callable[[cudf.DataFrame], cudf.DataFrame]] = None,
        fn_type: Literal["group", "block"] = "block",
    ) -> pa.Table:
        # extract the partitions from the shuffler and convert to cuDF DataFrames
        partitions = []
        for _, partition in self.extract():
            partitions.append(partition)

        # TODO: can this be done in a streaming manner?
        cdfs = [pylibcudf_to_cudf_dataframe(partition, self.columns) for partition in partitions]
        cdf = cudf.concat(cdfs) if len(cdfs) > 0 else cudf.DataFrame({col: [] for col in self.columns})

        # apply the map_groups_fn, if provided
        if map_groups_fn:
            if fn_type == "group":
                result = cdf.groupby(self.group_by).apply(map_groups_fn)
            elif fn_type == "block":
                result = map_groups_fn(cdf)
            else:
                raise ValueError(f"Invalid fn_type: {fn_type}")
        logger.info(f"Number of items in result: {len(result)}")
        return result.to_arrow()


class GPUDataset():
    """Wrapper for a ray.data.Dataset to perform GPU-accelerated shuffling via groupby().map_groups()."""
    def __init__(self, dataset: ray.data.Dataset, nranks: int):
        self.dataset = dataset
        self.nranks = nranks

    def groupby(self, key: Union[str, List[str], None], num_partitions: Optional[int] = None) -> "GPUDataset":
        self.key = key if isinstance(key, list) else [key]
        self.num_partitions = num_partitions if num_partitions else self.nranks

        # create the shuffle actors
        self.actors = [
            GPUShuffleActor.remote(
                nranks=self.nranks,
                hash_parallelism=self.num_partitions,
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
        logger.info(f"Actor pool setup complete")

        # insert dataset chunks into the actors
        batches = self.dataset.iter_batches(batch_size=1000*100, batch_format="pyarrow")
        batch_counts = list(self.pool.map(lambda actor, batch: actor.insert_batch.remote(batch), batches))
        logger.info(f"Total number of batches: {len(batch_counts)}")
        logger.info(f"Total number of items: {sum(batch_counts)}")
        logger.info(f"Actor pool insert chunks complete")

        # insert finished markers into the actors
        ray.get([actor.insert_finished.remote() for actor in self.actors])
        return self

    def map_groups(
        self,
        fn: Optional[Callable[[cudf.DataFrame], cudf.DataFrame]],
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
        # read the shuffled chunks from the actors (and apply map_groups_fn on each chunk)
        chunks = ray.get([actor.extract_partitions.remote(fn, fn_type) for actor in self.actors])
        logger.info(f"Number of chunks: {len(chunks)}")
        logger.info(f"Number of items in chunks: {sum([len(chunk) for chunk in chunks])}")
        logger.info(f"Actor pool read complete")

        # convert the list of pyarrow tables to a ray dataset
        self.dataset = ray.data.from_arrow(chunks)
        return self

    def materialize(self) -> ray.data.Dataset:
        return self.dataset.materialize()

    def join(
        self,
        right_ds,
        on,
        right_on=None,
        join_type='inner',
        num_partitions=None,
    ) -> "GPUDataset":
        """GPU-accelerated join implementation.

        Args:
            right_ds: Right dataset to join (can be Dataset or GPUDataset)
            on: Join key(s) for left dataset (string or list of strings)
            right_on: Join key(s) for right dataset (default: same as on)
            how: Join type - 'inner', 'left', or 'left_anti'
            num_partitions: Number of partitions for shuffle (default: nranks)

        Returns:
            GPUDataset wrapping the join result

        Example:
            result = left_ds.gpu(nranks=4).join(
                right_ds, on='id', how='inner'
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

        # Execute join
        executor = GPUJoinExecutor(
            left_ds=self.dataset,
            right_ds=right_ds,
            nranks=self.nranks,
            on=on_list,
            right_on=right_on_list,
            join_type=join_type,
            num_partitions=num_partitions or self.nranks,
        )

        result_ds = executor.execute()
        return GPUDataset(result_ds, nranks=self.nranks)


def dataset_to_gpu(self, nranks: Optional[int] = None):
    """Monkey-patch to convert ray.data.Dataset to GPUDataset for GPU-accelerated operations."""
    # Prevent double-wrapping
    if isinstance(self, GPUDataset):
        return self

    logger.debug(f"Converting Dataset to GPUDataset with nranks={nranks}")
    return GPUDataset(self, nranks=nranks)


def create_edges_from_collisions_gpu_block(cdf: cudf.DataFrame) -> cudf.DataFrame:
    """Create edges from a batch of candidate pairs that collide.

    This is a vectorized implementation, where the input includes all groups collected by the local shuffler."""
    cdf['count'] = 1
    grouped_df = (
        cdf.groupby(['band_id', 'band_hash'])
        .agg({'doc_id': 'min', 'count': 'sum'})
        .reset_index()
    )
    grouped_df.rename(columns={'doc_id': 'src', 'count': 'count'}, inplace=True)  # rename in place to avoid copy
    grouped_df = grouped_df.loc[grouped_df['count'] > 1]
    cdf = cdf.merge(grouped_df, on=['band_id', 'band_hash'], how='inner')
    cdf.rename(columns={'doc_id': 'dst'}, inplace=True)  # rename in place to avoid copy
    return cdf[['src', 'dst']]


def create_edges_from_collisions_gpu_group(cdf: cudf.DataFrame) -> cudf.DataFrame:
    """Create edges from a batch of candidate pairs that collide.

    This is a naive port of the CPU implementation, which is unusable due to very-poor GPU utilization/performance."""
    a = cdf['doc_id']
    n = a.shape[0]
    if n < 2:
        # Preserve dtype to maintain schema consistency across batches
        return cudf.DataFrame({'src': cudf.Series([], dtype=a.dtype), 'dst': cudf.Series([], dtype=a.dtype)})

    # indices for all i < j
    min_doc_id = a.min().item()
    src = cp.repeat(min_doc_id, n)
    dst = a
    return cudf.DataFrame({'src': src, 'dst': dst})


def create_edges_from_collisions(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Create edges from a batch of candidate pairs that collide."""
    # schema: band_id, band_hash, doc_id
    # all band_id and band_hash are the same
    # Generate all pairs of doc_id

    # batch['doc_id'] can be a pandas Series or 1D array-like
    a = np.asarray(batch['doc_id'])
    n = a.shape[0]
    if n < 2:
        # Preserve dtype to maintain schema consistency across batches
        return {'src': np.array([], dtype=a.dtype), 'dst': np.array([], dtype=a.dtype)}

    # indices for all i < j
    min_doc_id = min(a)
    src = np.repeat(min_doc_id, n)
    dst = a
    return {'src': src, 'dst': dst}


def shuffle_cpu(bands_ds: ray.data.Dataset, hash_parallelism: int):
    # Group by band to find candidate pairs
    edges_ds = (
        bands_ds.groupby(['band_id', 'band_hash'], num_partitions=hash_parallelism)
        .map_groups(create_edges_from_collisions, batch_format="numpy")
        .materialize()
    )
    return edges_ds


def shuffle_gpu(bands_ds: ray.data.Dataset, hash_parallelism: int, num_gpus: int):
    edges_ds = (
        GPUDataset(bands_ds, num_gpus).groupby(['band_id', 'band_hash'], num_partitions=hash_parallelism)
        .map_groups(create_edges_from_collisions_gpu_block, fn_type="block")
        # .map_groups(create_edges_from_collisions_gpu_group, fn_type="group")
        .materialize()
    )
    return edges_ds


def main(
    minhash_checkpoint_uri: str,
    edges_output: str,
    hash_parallelism: int = 100,
    num_gpus: int = 0,
):
    """Shuffle the minhash bands.

    Args:
        minhash_checkpoint_uri: URI of the minhash checkpoint.
        hash_parallelism: Number of partitions to use for the hash.
    Returns:
        edges_ds: Dataset of edges.
    """
    ray.init(num_gpus=num_gpus, _temp_dir=os.environ.get("RAY_TMP_DIR", "/tmp"))

    # Read the minhash checkpoint
    bands_ds = ray.data.read_parquet(minhash_checkpoint_uri)
    bands_ds = bands_ds.materialize()
    logger.info(f"Number of blocks in bands_ds: {bands_ds.num_blocks()}")

    # Group by band and hash to find candidate pairs
    start = time.time()
    if num_gpus == 0:
        edges_ds = shuffle_cpu(bands_ds, hash_parallelism)
    else:
        edges_ds = shuffle_gpu(bands_ds, hash_parallelism, num_gpus)

    edges_count = edges_ds.count()
    edges_ds.write_parquet(edges_output)
    stop = time.time()

    # edges_deduped = edges_ds.to_pandas().drop_duplicates()

    logger.info(f"Shuffle time: {stop - start} seconds")
    logger.info(f"Length of edges_ds: {edges_count}")
    # logger.info(f"Length of edges_deduped: {len(edges_deduped)}")


if __name__ == "__main__":
    fire.Fire(main)
