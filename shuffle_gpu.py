from typing import Dict

import logging
import time

import cudf
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
    def __init__(self, nranks: int, total_nparts: int, shuffle_on: list[str]):
        super().__init__(nranks=nranks, total_nparts=total_nparts, shuffle_on=shuffle_on)
        self.columns = ['doc_id', 'band_id', 'band_hash']
        logger.info(f"Rank {self.rank} setup complete")

    def insert_batch(self, batch: pa.Table) -> pa.Table:
        df = cudf.DataFrame.from_arrow(batch)
        # self.columns = list(df.columns)
        self.insert_chunk(table=df, column_names=self.columns)
        return len(batch)

    def extract_partitions(self) -> pa.Table:
        partitions = []
        for partition_id, partition in self.extract():
            partitions.append(partition)
        cdfs = [pylibcudf_to_cudf_dataframe(partition, self.columns) for partition in partitions]
        cdf = cudf.concat(cdfs)
        result = cdf.to_arrow()
        return result

def create_edges_from_collisions_gpu(batch: pa.Table) -> pa.Table:
    """Create edges from a batch of candidate pairs that collide."""
    df = cudf.DataFrame.from_arrow(batch)
    df['count'] = 1
    grouped_df = (
        df.groupby(['band_id', 'band_hash'])
        .agg({'doc_id': 'min', 'count': 'sum'})
        .reset_index().rename(columns={'doc_id': 'src', 'count': 'count'})
    )
    grouped_df = grouped_df.loc[grouped_df['count'] > 1]
    df = df.merge(grouped_df, on=['band_id', 'band_hash'], how='inner').rename(columns={'doc_id': 'dst'})
    return df[['src', 'dst']].to_arrow()

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
    # pre-create the shuffle actors to group by band and hash
    actors = [
        GPUShuffleActor.remote(nranks=num_gpus, total_nparts=hash_parallelism, shuffle_on=['band_id', 'band_hash'])
        for _ in range(num_gpus)
    ]
    _, root_address = ray.get(actors[0].setup_root.remote())
    ray.get([actor.setup_worker.remote(root_address) for actor in actors])
    pool = ActorPool(actors)
    logger.info(f"Actor pool setup complete")

    # insert chunks into the actors
    batches = bands_ds.iter_batches(batch_size=1000*100, batch_format="pyarrow")
    batch_counts = list(pool.map(lambda actor, batch: actor.insert_batch.remote(batch), batches))
    logger.info(f"Total number of batches: {len(batch_counts)}")
    logger.info(f"Total number of items: {sum(batch_counts)}")
    logger.info(f"Actor pool insert chunks complete")

    # insert finished markers into the actors
    ray.get([actor.insert_finished.remote() for actor in actors])
    logger.info(f"Actor pool insert finished complete")

    # read the chunks from the actors
    chunks = ray.get([actor.extract_partitions.remote() for actor in actors])
    logger.info(f"Actor pool read complete")

    # convert the chunks to a dataset
    # TODO: move map_batches to shuffle_gpu to avoid unnecessary data movement
    shuffled_ds = ray.data.from_arrow(chunks)
    shuffled_ds.write_parquet("shuffled_gpu.parquet")

    edges_ds = shuffled_ds.map_batches(
        create_edges_from_collisions_gpu,
        batch_format="pyarrow",
        batch_size=1000*100,
        num_gpus=1,
    )

    return edges_ds


def main(
    minhash_checkpoint_uri: str,
    edges_output: str,
    hash_parallelism: int = 100,
    num_gpus: int = 0,
    limit: int = None,
):
    """Shuffle the minhash bands.

    Args:
        minhash_checkpoint_uri: URI of the minhash checkpoint.
        hash_parallelism: Number of partitions to use for the hash.
    Returns:
        edges_ds: Dataset of edges.
    """
    ray.init(num_gpus=num_gpus)

    # Read the minhash checkpoint
    bands_ds = ray.data.read_parquet(minhash_checkpoint_uri)
    if limit is not None:
        bands_ds = bands_ds.limit(limit)
    bands_ds = bands_ds.materialize()
    logger.info(f"Number of blocks in bands_ds: {bands_ds.num_blocks()}")

    # Shuffle the minhash bands using CPU object store
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