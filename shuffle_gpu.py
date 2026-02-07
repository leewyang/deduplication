from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Union

import os
import logging
import time

import cudf
import cupy as cp
import fire
import numpy as np
import ray

import gpu_dataset  # noqa: F401, monkey-patch ray.data.Dataset


logging.basicConfig(
    format='%(asctime)s %(levelname)s %(name)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


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
    if num_gpus > 0:
        edges_ds = (
            bands_ds.gpu(nranks=num_gpus)
            .groupby(['band_id', 'band_hash'], num_partitions=hash_parallelism)
            .map_groups(create_edges_from_collisions_gpu_block, fn_type="block")
            .materialize()
        )
    else:
        edges_ds = (
            bands_ds.groupby(['band_id', 'band_hash'], num_partitions=hash_parallelism)
            .map_groups(create_edges_from_collisions, batch_format="numpy")
            .materialize()
        )

    edges_count = edges_ds.count()
    edges_ds.write_parquet(edges_output)
    stop = time.time()

    # edges_deduped = edges_ds.to_pandas().drop_duplicates()

    logger.info(f"Shuffle time: {stop - start} seconds")
    logger.info(f"Length of edges_ds: {edges_count}")
    # logger.info(f"Length of edges_deduped: {len(edges_deduped)}")


if __name__ == "__main__":
    fire.Fire(main)
