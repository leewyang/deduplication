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
        cdf = cudf.concat(cdfs)

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
        self.key = key
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
