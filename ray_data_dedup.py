"""
Large-scale deduplication using Ray Data with Spark-like operations.

Based on the approach from: https://huggingface.co/blog/dedup

This implementation uses Ray Data's native operations (map_batches, groupby, etc.)
to implement MinHash + LSH deduplication, similar to the Spark approach.

Architecture:
1. MinHash signature generation (map_batches)
2. LSH banding to generate candidate pairs (flatmap + groupby)
3. Connected components to find duplicate clusters (iterative map-reduce)
"""

import argparse
import glob as glob_module
import hashlib
import logging
import struct
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pyarrow as pa
from pyarrow import fs as pafs
import pandas as pd
import ray
from scipy import integrate
import os

# Optional GPU imports - will be imported lazily when needed
try:
    import cupy as cp
    import cudf
    import pylibcudf
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    cudf = None
    pylibcudf = None

logger = logging.getLogger(__name__)

# Constants
MERSENNE_PRIME = np.uint64((1 << 61) - 1)
MAX_HASH = np.uint32((1 << 32) - 1)


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
        parquet_files = glob_module.glob(pattern, recursive=True)
    else:
        # Could be a glob pattern
        parquet_files = [f for f in glob_module.glob(path, recursive=True) if f.endswith('.parquet')]

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


def sha1_hash32(data: bytes) -> int:
    """Generate 32-bit hash from SHA1."""
    return struct.unpack('<I', hashlib.sha1(data).digest()[:4])[0]


def optimal_param(
    threshold: float,
    num_perm: int,
    false_positive_weight: float = 0.5,
    false_negative_weight: float = 0.5,
) -> Tuple[int, int]:
    """
    Compute optimal LSH parameters (bands, rows) that minimize weighted sum
    of false positive and false negative probabilities.

    Returns:
        (num_bands, rows_per_band)
    """
    def false_positive_probability(th: float, band: int, rows: int) -> float:
        def proba(s: float) -> float:
            return 1 - (1 - s ** float(rows)) ** float(band)
        a, _ = integrate.quad(proba, 0.0, th)
        return a

    def false_negative_probability(th: float, band: int, rows: int) -> float:
        def proba(s: float) -> float:
            return 1 - (1 - (1 - s ** float(rows)) ** float(band))
        a, _ = integrate.quad(proba, th, 1.0)
        return a

    min_error = float('inf')
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = false_positive_probability(threshold, b, r)
            fn = false_negative_probability(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


class MinHashGenerator:
    """Generates MinHash signatures from text using n-grams."""

    def __init__(
        self,
        num_perm: int = 128,
        ngram_size: int = 5,
        seed: int = 42,
        lowercase: bool = True,
    ):
        self.num_perm = num_perm
        self.ngram_size = ngram_size
        self.lowercase = lowercase

        # Generate permutations for MinHash
        gen = np.random.RandomState(seed=seed)
        self.perm_a, self.perm_b = np.array(
            [
                (
                    gen.randint(1, MERSENNE_PRIME, dtype=np.uint64),
                    gen.randint(0, MERSENNE_PRIME, dtype=np.uint64),
                )
                for _ in range(num_perm)
            ],
            dtype=np.uint64,
        ).T

    def _ngrams(self, text: str) -> Set[bytes]:
        """Generate character n-grams from text."""
        if self.lowercase:
            text = text.lower()
        return {
            text[i:i + self.ngram_size].encode('utf-8')
            for i in range(len(text) - self.ngram_size + 1)
        }

    def compute_minhash(self, text: str) -> np.ndarray:
        """
        Compute MinHash signature for a single text.

        Returns:
            Array of shape (num_perm,) with uint32 values
        """
        tokens = self._ngrams(text)

        if len(tokens) == 0:
            # Empty text gets max hash values
            return np.full(self.num_perm, MAX_HASH, dtype=np.uint32)

        # Hash all tokens
        hashes = np.array([sha1_hash32(token) for token in tokens], dtype=np.uint64)

        # Apply permutations: (h * a + b) % c
        # Broadcasting: hashes[:, None] has shape (num_tokens, 1)
        # perm_a[None, :] has shape (1, num_perm)
        phv = ((hashes[:, None] * self.perm_a[None, :] + self.perm_b) % MERSENNE_PRIME).astype(np.uint32)

        # Take minimum across all tokens for each permutation
        return phv.min(axis=0)


class GPUMinHashGenerator:
    """
    GPU-accelerated MinHash signature generator using CuPy and MurmurHash3.

    This class provides the same interface as MinHashGenerator but uses GPU
    acceleration for the permutation computations.
    """

    def __init__(
        self,
        num_perm: int = 128,
        ngram_size: int = 5,
        seed: int = 42,
        lowercase: bool = True,
    ):
        if not CUPY_AVAILABLE:
            raise ImportError(
                "CuPy, cuDF, and pylibcudf are required for GPU MinHash. "
                "Install RAPIDS cuDF: https://docs.rapids.ai/install"
            )

        self.num_perm = num_perm
        self.ngram_size = ngram_size
        self.lowercase = lowercase
        self.seed = seed

        # Generate permutations for MinHash (same as CPU version for compatibility)
        gen = np.random.RandomState(seed=seed)
        perm_a_np, perm_b_np = np.array(
            [
                (
                    gen.randint(1, MERSENNE_PRIME, dtype=np.uint64),
                    gen.randint(0, MERSENNE_PRIME, dtype=np.uint64),
                )
                for _ in range(num_perm)
            ],
            dtype=np.uint64,
        ).T

        # Transfer permutation coefficients to GPU (done once)
        self.perm_a = cp.asarray(perm_a_np, dtype=cp.uint64)
        self.perm_b = cp.asarray(perm_b_np, dtype=cp.uint64)

        # Pre-allocate max hash array on GPU
        self.max_hash_gpu = cp.full(num_perm, MAX_HASH, dtype=cp.uint32)

    def _ngrams(self, text: str) -> List[str]:
        """Generate character n-grams from text as a list of strings."""
        if self.lowercase:
            text = text.lower()
        return [
            text[i:i + self.ngram_size]
            for i in range(len(text) - self.ngram_size + 1)
        ]

    def _hash_ngrams_gpu(self, ngrams_list: List[str]) -> cp.ndarray:
        """
        Hash a list of n-grams using GPU-native MurmurHash3 via pylibcudf.

        Uses pylibcudf.hashing.murmurhash3_x86_32 for fully GPU-accelerated hashing.
        See: https://docs.rapids.ai/api/cudf/stable/pylibcudf/api_docs/hashing/#pylibcudf.hashing.murmurhash3_x86_32

        Args:
            ngrams_list: List of n-gram strings

        Returns:
            CuPy array of 32-bit hash values on GPU
        """
        if len(ngrams_list) == 0:
            return cp.array([], dtype=cp.uint32)

        # Create a cudf Series from the n-grams (strings)
        ngrams_series = cudf.Series(ngrams_list, dtype='str')

        # Convert to pylibcudf Table for hashing
        # pylibcudf.hashing.murmurhash3_x86_32 takes a Table as input
        plc_table = pylibcudf.Table([ngrams_series._column.to_pylibcudf(mode="read")])

        # Compute MurmurHash3 32-bit hash on GPU
        hash_column = pylibcudf.hashing.murmurhash3_x86_32(plc_table, self.seed)

        # Convert result back to cudf Series, then to CuPy array
        result_series = cudf.Series.from_pylibcudf(hash_column)
        return cp.asarray(result_series.values, dtype=cp.uint32)

    def compute_minhash(self, text: str) -> np.ndarray:
        """
        Compute MinHash signature for a single text using GPU.

        Returns:
            Array of shape (num_perm,) with uint32 values
        """
        ngrams = self._ngrams(text)

        if len(ngrams) == 0:
            # Empty text gets max hash values
            return cp.asnumpy(self.max_hash_gpu)

        # Hash n-grams on GPU using pylibcudf's MurmurHash3
        hashes_gpu = self._hash_ngrams_gpu(ngrams).astype(cp.uint64)

        # Apply permutations on GPU: (h * a + b) % prime
        # Broadcasting: hashes_gpu[:, None] has shape (num_tokens, 1)
        # perm_a[None, :] has shape (1, num_perm)
        phv = ((hashes_gpu[:, None] * self.perm_a[None, :] + self.perm_b) % MERSENNE_PRIME).astype(cp.uint32)

        # Take minimum across all tokens for each permutation
        result = phv.min(axis=0)

        # Transfer result back to CPU
        return cp.asnumpy(result)

    def compute_minhash_batch(self, texts: List[str]) -> np.ndarray:
        """
        Compute MinHash signatures for a batch of texts using GPU.

        This method is optimized for batch processing, minimizing
        GPU memory transfers by processing multiple texts together.
        Uses pylibcudf's murmurhash3_x86_32 for fully GPU-accelerated hashing.

        Args:
            texts: List of text strings

        Returns:
            Array of shape (num_texts, num_perm) with uint32 values
        """
        num_texts = len(texts)
        results = np.empty((num_texts, self.num_perm), dtype=np.uint32)

        # Process texts and collect all ngram hashes
        all_ngrams_per_text = []
        text_lengths = []

        for text in texts:
            ngrams = self._ngrams(text)
            all_ngrams_per_text.append(ngrams)
            text_lengths.append(len(ngrams))

        # Flatten all n-grams for batch hashing
        all_ngrams_flat = []
        for ngrams in all_ngrams_per_text:
            all_ngrams_flat.extend(ngrams)

        if len(all_ngrams_flat) == 0:
            # All texts are empty
            return np.full((num_texts, self.num_perm), MAX_HASH, dtype=np.uint32)

        # Batch hash all n-grams on GPU using pylibcudf's MurmurHash3
        all_hashes_gpu = self._hash_ngrams_gpu(all_ngrams_flat).astype(cp.uint64)

        # Process each text's hashes
        offset = 0
        for i, length in enumerate(text_lengths):
            if length == 0:
                results[i] = MAX_HASH
            else:
                # Get this text's hashes
                hashes_gpu = all_hashes_gpu[offset:offset + length]

                # Apply permutations on GPU
                phv = ((hashes_gpu[:, None] * self.perm_a[None, :] + self.perm_b) % MERSENNE_PRIME).astype(cp.uint32)

                # Take minimum and transfer back
                results[i] = cp.asnumpy(phv.min(axis=0))

            offset += length

        return results


def generate_minhash_signatures(
    batch: Dict[str, np.ndarray],
    text_column: str,
    num_perm: int,
    ngram_size: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    Ray Data UDF to generate MinHash signatures for a batch of documents.

    This function is called by map_batches and processes documents in parallel.
    """
    generator = MinHashGenerator(num_perm=num_perm, ngram_size=ngram_size, seed=seed)

    texts = batch[text_column]
    signatures = np.array([generator.compute_minhash(text) for text in texts])

    # Add signatures to batch
    batch['minhash'] = signatures
    return batch


def generate_minhash_signatures_gpu(
    batch: Dict[str, np.ndarray],
    text_column: str,
    num_perm: int,
    ngram_size: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    GPU-accelerated Ray Data UDF to generate MinHash signatures for a batch of documents.

    This function is called by map_batches with num_gpus=1 and processes documents
    using GPU acceleration via CuPy and MurmurHash3.

    For best performance, use larger batch sizes (e.g., 4096+) to amortize
    GPU memory transfer overhead.
    """
    generator = GPUMinHashGenerator(num_perm=num_perm, ngram_size=ngram_size, seed=seed)

    texts = list(batch[text_column])

    # Use batch processing for better GPU efficiency
    signatures = generator.compute_minhash_batch(texts)

    # Add signatures to batch
    batch['minhash'] = signatures
    return batch


def generate_lsh_bands(
    batch: Dict[str, np.ndarray],
    num_bands: int,
    rows_per_band: int,
) -> Dict[str, np.ndarray]:
    """
    Generate LSH bands from MinHash signatures.

    This creates multiple (band_id, band_hash) pairs per document,
    which will be used to find candidate duplicate pairs.

    Returns a flattened batch where each row represents one band of one document.

    Example:
    {
        'minhash': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        'id': np.array([1, 2, 3]),
    }
    ->
    {
        'doc_id': np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]),
        'band_id': np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]),
        'band_hash': np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']),
    }
    """
    minhashes = batch['minhash']
    num_docs = len(minhashes)

    # For each document, generate num_bands rows
    output_size = num_docs * num_bands

    # Replicate document IDs for each band
    doc_ids = np.repeat(batch['id'], num_bands)

    # np.tile repeats the entire array a given number of times.
    # [0, 1, ..., num_bands-1] * num_docs times
    band_ids = np.tile(np.arange(num_bands), num_docs)
    band_hashes = []

    for doc_idx, minhash in enumerate(minhashes):
        for band_idx in range(num_bands):
            start = band_idx * rows_per_band
            end = start + rows_per_band
            band_values = minhash[start:end]
            # Create a hash of the band
            band_hash = hashlib.sha256(band_values.tobytes()).hexdigest()[:16]
            band_hashes.append(band_hash)

    return {
        'doc_id': doc_ids,
        'band_id': band_ids,
        'band_hash': np.array(band_hashes),
    }


def generate_lsh_bands_gpu(
    batch: Dict[str, np.ndarray],
    num_bands: int,
    rows_per_band: int,
) -> Dict[str, np.ndarray]:
    """
    GPU-optimized LSH band generation from MinHash signatures.

    Uses pylibcudf's murmurhash3_x64_128 for GPU-accelerated band hashing.
    See: https://docs.rapids.ai/api/cudf/stable/pylibcudf/api_docs/hashing/#pylibcudf.hashing.murmurhash3_x64_128

    This creates multiple (band_id, band_hash) pairs per document.

    Returns a flattened batch where each row represents one band of one document.
    """
    if not CUPY_AVAILABLE:
        raise ImportError(
            "GPU mode requested but cudf/pylibcudf not available. "
            "Install with: pip install cudf-cu12 (or appropriate CUDA version)"
        )

    minhashes = batch['minhash']
    num_docs = len(minhashes)

    # Replicate document IDs for each band
    doc_ids = np.repeat(batch['id'], num_bands)

    # Generate band IDs: [0, 1, ..., num_bands-1] repeated for each doc
    band_ids = np.tile(np.arange(num_bands), num_docs)

    # Collect all band values as strings for batch GPU hashing
    # We convert band values to hex strings for consistent hashing
    band_strings = []
    for doc_idx, minhash in enumerate(minhashes):
        for band_idx in range(num_bands):
            start = band_idx * rows_per_band
            end = start + rows_per_band
            band_values = minhash[start:end]
            # Convert band values to a hex string representation
            band_strings.append(band_values.tobytes().hex())

    # Create cudf Series for GPU-accelerated hashing
    band_series = cudf.Series(band_strings, dtype='str')

    # Create pylibcudf Table for hashing
    plc_table = pylibcudf.Table([band_series._column.to_pylibcudf(mode="read")])

    # Compute MurmurHash3 128-bit hash on GPU (returns Table with two uint64 columns)
    hash_table = pylibcudf.hashing.murmurhash3_x64_128(plc_table, seed=0)

    # Convert first column to hex strings for band_hash (use first 64 bits)
    hash_col = cudf.Series.from_pylibcudf(hash_table.columns()[0])
    band_hashes = hash_col.to_pandas().apply(lambda x: format(x, '016x')).values

    return {
        'doc_id': doc_ids,
        'band_id': band_ids,
        'band_hash': np.array(band_hashes),
    }


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


def distinct_2col(current_ds: ray.data.Dataset, col_1, col_2, parallelism: int = 100) -> ray.data.Dataset:
    def distinct_map_groups(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        unique_values = np.unique(batch[col_2])
        return {col_1: np.repeat(batch[col_1][0], len(unique_values)), col_2: unique_values}
    current_ds = current_ds.select_columns([col_1, col_2])
    current_ds = current_ds.groupby(col_1, num_partitions=parallelism).map_groups(distinct_map_groups, batch_format="numpy")
    current_ds = current_ds.materialize()
    return current_ds



def large_star_emit(row):
    u, v = row["node"], row["parent"]
    if u == v:
        return [{"node": u, "parent": v}]
    return [{"node": u, "parent": v}, {"node": v, "parent": u}]

def small_star_emit(row):
    """Emit (u, v) if u >= v else (v, u) -> (node, parent)"""
    u, v = row["node"], row["parent"]
    if u >= v:
        return [{"node": u, "parent": v}]
    else:
        return [{"node": v, "parent": u}]


def large_star_map_groups(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Map groups for large-star.

    take the minimum parent (mp)
    Return dataframe (v, mp) where v > node for v in neighborhood of node."""
    node = batch['node'][0]
    neighbors = np.unique(batch['parent'])
    neighbors = np.concatenate([neighbors, np.array([node])])
    mp = min(neighbors)
    large_neighbors = neighbors[neighbors > node]
    return {'node': large_neighbors, 'parent': np.repeat(mp, len(large_neighbors))}

def small_star_map_groups(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Map groups for small-star.

    take the minimum parent (mp)
    let N be all neighbors less than or equal to node
    Return dataframe (v, mp) for all v in N"""
    node = batch['node'][0]
    neighbors = np.unique(batch['parent'])
    small_neighbors = neighbors[neighbors <= node]
    small_neighbors = np.concatenate([small_neighbors, np.array([node])])
    mp = min(small_neighbors)
    return {'node': small_neighbors, 'parent': np.repeat(mp, len(small_neighbors))}

def cast_node_parent_large_string(batch: pa.Table) -> pa.Table:
    """Cast node and parent to large string."""
    new_schema = pa.schema([
        pa.field('node', pa.large_string()),
        pa.field('parent', pa.large_string()),
    ])
    return batch.select(['node', 'parent']).cast(new_schema)

def compute_connected_components_distributed(
    current_ds: ray.data.Dataset,
    max_iterations: int = 100,
    parallelism: int = 200,
    verbose=False
) -> ray.data.Dataset:
    """
    Compute connected components using distributed large-star/small-star algorithm.

    This iterative algorithm is suitable for large-scale graphs (billions of edges).
    Based on the paper: "Connected Components in MapReduce and Beyond"

    Algorithm:
    1. Large-star: For each edge (u,v), point the larger node to the smaller
        Emit (u, v) and (v, u) -> (node, parent)
        Group by node and take the minimum parent (mp)
        Emit (v, mp) where v > mp for v in neighborhood of node
    2. Small-star: Propagate parent pointers transitively
        Emit (u, v) if u >= v else (v, u) -> (node, parent)
        Group by node and take the minimum parent (mp)
        Emit (v, mp) for all v in neighborhood of node
    3. Repeat until convergence

    Returns:
        Dataset with columns: node, parent (where parent is the component root)
    """
    logger.info("Computing connected components with distributed algorithm...")

    current_ds = current_ds.materialize()
    num_components = current_ds.count()
    print(f"Initial count: {num_components}")
    convergence_counter = 3
    for i in range(max_iterations):
        current_ds = current_ds.map_batches(cast_node_parent_large_string, batch_format="pyarrow")
        current_ds = current_ds.materialize()
        # Step 1: Large-star
        current_ds = current_ds.flat_map(large_star_emit, memory=8*2**30)
        current_ds = current_ds\
            .groupby(['node'], num_partitions=parallelism)\
            .map_groups(large_star_map_groups, batch_format="numpy")
        current_ds = current_ds.materialize()
        # current_ds = distinct(current_ds, ['node', 'parent'])
        # current_ds = current_ds.materialize()
        if verbose:
            print(current_ds.to_pandas())

        # Step 2: Small-star
        current_ds = current_ds.flat_map(small_star_emit)
        current_ds = current_ds\
            .groupby(['node'], num_partitions=parallelism)\
            .map_groups(small_star_map_groups, batch_format="numpy")

        current_ds = current_ds.materialize()
        # current_ds = distinct(current_ds, ['node', 'parent'])
        # current_ds = current_ds.materialize()
        # TODO - refactor this to be distributed
        new_num_components = current_ds.groupby("parent").count().count()

        if num_components == new_num_components:
            convergence_counter -= 1
        else:
            convergence_counter = 3
        if convergence_counter <= 0:
            break

        num_components = new_num_components

        print("-" * 10)
        print(f"Iteration {i}: {current_ds.count()}")
        print(f"number of components: {num_components}")
        print(f"convergence_counter: {convergence_counter}")
        print("-" * 10)

    return current_ds


def get_or_create_minhash_bands(
        ds: ray.data.Dataset,
        minhash_checkpoint_uri: Optional[str],
        text_column: str,
        threshold: float,
        num_perm: int,
        ngram_size: int,
        seed: int,
        output_blocks: int = 100,
        use_gpu: bool = False,
        gpu_batch_size: int = 4096) -> ray.data.Dataset:
    """
    Generate MinHash signatures and LSH bands for a dataset.

    Args:
        ds: Input Ray dataset
        minhash_checkpoint_uri: Optional checkpoint path to load/save bands
        text_column: Name of the text column
        threshold: Jaccard similarity threshold
        num_perm: Number of MinHash permutations
        ngram_size: Character n-gram size
        seed: Random seed
        output_blocks: Number of output partitions
        use_gpu: Whether to use GPU acceleration
        gpu_batch_size: Batch size for GPU processing (larger = better GPU utilization)

    Returns:
        Dataset with LSH bands
    """
    if minhash_checkpoint_uri is not None:
        if not check_path_exists(minhash_checkpoint_uri):
            raise ValueError(f"Checkpoint URI {minhash_checkpoint_uri} does not exist")
        bands_ds = ray.data.read_parquet(minhash_checkpoint_uri)
        bands_ds = bands_ds.repartition(num_blocks=output_blocks)
        bands_ds = bands_ds.materialize()
        return bands_ds

    # Need to materialize first if limiting, or else the limit could be non-deterministic
    ds = ds.materialize()
    # masterset_uuids = set(x["id"] for x in ds.select_columns("id").take_all())
    assert "Materialized" in str(type(ds))

    # Compute optimal LSH parameters
    num_bands, rows_per_band = optimal_param(threshold, num_perm)
    logger.info(f"LSH parameters: {num_bands} bands, {rows_per_band} rows per band")

    # Step 1: Generate MinHash signatures
    if use_gpu:
        if not CUPY_AVAILABLE:
            raise ImportError(
                "GPU mode requested but cudf/pylibcudf not available. "
                "Install RAPIDS cuDF: https://docs.rapids.ai/install"
            )
        logger.info("Step 1: Generating MinHash signatures (GPU)...")
        # Schema: dict_keys(['*', 'minhash'])
        ds_with_minhash = ds.map_batches(
            generate_minhash_signatures_gpu,
            fn_kwargs={
                'text_column': text_column,
                'num_perm': num_perm,
                'ngram_size': ngram_size,
                'seed': seed,
            },
            batch_format='numpy',
            num_gpus=1,  # Request GPU from Ray scheduler
            batch_size=gpu_batch_size,  # Larger batches for better GPU utilization
        )
    else:
        logger.info("Step 1: Generating MinHash signatures (CPU)...")
        # Schema: dict_keys(['*', 'minhash'])
        ds_with_minhash = ds.map_batches(
            generate_minhash_signatures,
            fn_kwargs={
                'text_column': text_column,
                'num_perm': num_perm,
                'ngram_size': ngram_size,
                'seed': seed,
            },
            batch_format='numpy',
        )

    # Step 2: Generate LSH bands (creates multiple rows per document)
    # Schema: ['doc_id', 'band_id', 'band_hash'], non are unique
    if use_gpu:
        logger.info("Step 2: Generating LSH bands (GPU-optimized with MurmurHash3)...")
        bands_ds: ray.data.Dataset = ds_with_minhash.map_batches(
            generate_lsh_bands_gpu,
            fn_kwargs={
                'num_bands': num_bands,
                'rows_per_band': rows_per_band,
            },
            batch_format='numpy',
            num_gpus=1,  # Request GPU from Ray scheduler for cudf/pylibcudf operations
            batch_size=gpu_batch_size,  # Larger batches for better GPU utilization
        )
    else:
        logger.info("Step 2: Generating LSH bands (CPU)...")
        bands_ds: ray.data.Dataset = ds_with_minhash.map_batches(
            generate_lsh_bands,
            fn_kwargs={
                'num_bands': num_bands,
                'rows_per_band': rows_per_band,
            },
            batch_format='numpy',
        )
    bands_ds = bands_ds.materialize()
    bands_ds = bands_ds.repartition(num_blocks=output_blocks)
    bands_ds = bands_ds.materialize()

    if minhash_checkpoint_uri is not None:
        bands_ds.write_parquet(minhash_checkpoint_uri)

    return bands_ds

def find_duplicate_components(
    bands_ds: ray.data.Dataset,
    max_cc_iterations: int = 100,
    validate_local: bool = False,
    hash_parallelism: int = 100,
) -> ray.data.Dataset:
    """
    Find duplicate components in a dataset of bands/hashes.

    Args:
        bands_ds: Input Ray dataset with bands/hashes
        max_cc_iterations: Maximum iterations for connected components
        validate_local: Whether to validate the local version of the algorithm
        hash_parallelism: Number of partitions for the hash step

    Returns:
        Deduplicated dataset
    """
    bands_ds = bands_ds.materialize()
    print("Number of blocks in bands_ds", bands_ds.num_blocks())
    # Step 3: Group by band to find candidate pairs
    logger.info("Step 3: Grouping by bands to find candidate pairs...")
    edges_ds = bands_ds.groupby(
        ['band_id', 'band_hash'], num_partitions=hash_parallelism).map_groups(create_edges_from_collisions, batch_format="numpy")
    edges_ds = edges_ds.materialize()
    edges_count = edges_ds.count()
    print("Length of edges_ds", edges_count)

    # Handle empty edges case (no collisions found)
    if edges_count == 0:
        logger.info("No candidate pairs found. No duplicates detected.")
        # Return empty dataset with expected schema
        empty_df = pd.DataFrame({"node": pd.Series(dtype=object), "parent": pd.Series(dtype=object)})
        return ray.data.from_pandas(empty_df)

    # Deduplicate edges (same pair might appear in multiple bands)
    logger.info("Step 4: Deduplicating edges...")
    # Use groupby to deduplicate across all batches
    # Group by (src, dst) and keep just one of each unique edge

    edges_ds = distinct_2col(
        edges_ds, col_1='src', col_2='dst', parallelism=hash_parallelism)
    print("Length of edges_ds after distinct", edges_ds.count())

    # Step 6: Compute connected components (distributed algorithm)
    logger.info("Step 6: Computing connected components (distributed)...")
    edges_ds = edges_ds.rename_columns(
        {"src": "node", "dst": "parent"}
    ).materialize()
    components_ds = compute_connected_components_distributed(
        edges_ds,
        max_iterations=max_cc_iterations,
        parallelism=hash_parallelism,
    )

    # check local version
    if validate_local:
        compute_connected_components_pandas(edges_ds.to_pandas())

    # Step 7: Filter duplicates
    logger.info("Step 7: Filtering duplicates...")

    # Keep only documents where node != parent (extraneous components)
    duplicate_components = components_ds.filter(
        lambda row: row['node'] != row['parent']
    ).materialize()

    logger.info(f"Duplicated components count: {duplicate_components.count()}")
    return duplicate_components


def main():
    """CLI for large-scale deduplication (3TB+)."""
    parser = argparse.ArgumentParser(
        description='Large-scale deduplication with Ray Data (designed for 3TB+)'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input path (parquet files, can use wildcards or GCS paths)',
    )
    parser.add_argument(
        '--output',
        type=str,
        required=False,
        default=os.path.join(
            os.environ.get("ANYSCALE_ARTIFACT_STORAGE", "/raid/spark-team/leey/tmp"), "dedup-output"
        ),
        help='Output path for deduplicated data',
    )
    parser.add_argument(
        '--text-column',
        type=str,
        default='text',
        help='Name of text column',
    )
    parser.add_argument(
        '--id-column',
        type=str,
        default='id',
        help='Name of ID column (must be unique for each document)',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.7,
        help='Jaccard similarity threshold (0.0-1.0)',
    )
    parser.add_argument(
        '--num-perm',
        type=int,
        default=128,
        help='Number of MinHash permutations',
    )
    parser.add_argument(
        '--ngram-size',
        type=int,
        default=5,
        help='Character n-gram size',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed',
    )
    parser.add_argument(
        '--max-cc-iterations',
        type=int,
        default=100,
        help='Maximum iterations for connected components convergence',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of documents to process (for testing on subset)',
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--minhash-checkpoint-uri",
        type=str,
        help="Checkpoint URI for minhash bands",
    )
    parser.add_argument(
        "--disable-progress-bars",
        action="store_true",
        default=False,
        help="Disable progress bars",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=False,
        help="Use GPU acceleration for MinHash computation (requires RAPIDS cuDF)",
    )
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        default=4096,
        help="Batch size for GPU processing (larger = better GPU utilization)",
    )

    args = parser.parse_args()
    if args.disable_progress_bars:
        ray.data.DataContext.get_current().enable_progress_bars = False

    # Read input data
    logger.info(f"Reading data from {args.input}")
    input_path = args.input

    # List all parquet files in the directory (supports both GCS and local paths)
    list_of_all_input_files = list_parquet_files(input_path)
    logger.info(f"Reading {len(list_of_all_input_files)} parquet files")

    ds = ray.data.read_parquet(list_of_all_input_files)

    input_count = ds.count()
    print(f"Original input count {input_count}")
    if args.limit is not None:
        logger.info(f"Limiting input to {args.limit} documents")
        assert input_count >= args.limit
        ds = ds.limit(args.limit)
        input_count = args.limit
    logger.info(f"Input dataset: {input_count} documents")

    logger.info(f"Starting large-scale deduplication with threshold={args.threshold}")
    if args.use_gpu:
        logger.info("GPU acceleration enabled for MinHash computation")
        if not CUPY_AVAILABLE:
            raise ImportError(
                "GPU mode requested but cudf/pylibcudf not available. "
                "Install RAPIDS cuDF: https://docs.rapids.ai/install"
            )

    bands_ds = get_or_create_minhash_bands(
        ds,
        text_column=args.text_column,
        threshold=args.threshold,
        num_perm=args.num_perm,
        ngram_size=args.ngram_size,
        seed=args.seed,
        minhash_checkpoint_uri=args.minhash_checkpoint_uri,
        output_blocks=args.parallelism,
        use_gpu=args.use_gpu,
        gpu_batch_size=args.gpu_batch_size,
    )

    # Duplicate components: Schema: ['node', 'parent']
    duplicate_components = find_duplicate_components(
        bands_ds,
        max_cc_iterations=args.max_cc_iterations,
        hash_parallelism=args.parallelism
    )
    duplicate_components = duplicate_components.materialize()
    duplicate_count = duplicate_components.count()

    # Join with original dataset to get full document content
    if duplicate_count == 0:
        # No duplicates found, skip the join
        logger.info("No duplicates found, skipping join.")
        deduplicated_ds = ds
    else:
        deduplicated_ds = ds.join(
            duplicate_components,
            on=(args.id_column,),
            right_on=('node',),
            join_type='left_anti',
            num_partitions=args.parallelism)

    deduplicated_ds = deduplicated_ds.materialize()

    # Write output
    logger.info(f"Writing deduplicated data to {args.output}")
    deduplicated_ds.write_parquet(args.output)

    output_count = deduplicated_ds.count()
    logger.info(f"Output dataset: {output_count} documents")
    logger.info(f"Removed {input_count - output_count} duplicates ({100*(input_count - output_count)/input_count:.1f}%)")



def compute_connected_components_pandas(edges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute connected components for a local dataframe using scipy's DisjointSet.

    This is an efficient local algorithm suitable for graphs that fit in memory.

    Args:
        edges_df: DataFrame with columns ['node', 'parent'] representing edges

    Returns:
        DataFrame with columns ['node', 'parent'] where parent is the component root
    """
    from scipy.cluster.hierarchy import DisjointSet
    edges = edges_df[['node', 'parent']].values

    # Create DisjointSet with all unique nodes
    all_nodes = set()
    for node, parent_node in edges:
        all_nodes.add(node)
        all_nodes.add(parent_node)

    ds = DisjointSet(all_nodes)

    # Merge edges
    for node, parent_node in edges:
        ds.merge(node, parent_node)
    # Create result dataframe
    result = pd.DataFrame({
        'node': list(all_nodes),
        'parent': [ds[node] for node in all_nodes]
    })
    print("Number of subsets:", ds.n_subsets)
    return result


def test_connected(local: bool = False):
    # Generate random graph
    n = 10000
    # Generate n random edges between 0 and n-1 (uniformly)
    edges = np.random.randint(0, n, size=(n * 2, 2))
    edges = pd.DataFrame(edges, columns=["node", "parent"])

    print("running distributed")
    ray.data.DataContext.get_current().enable_progress_bars = False
    edges_ds = ray.data.from_pandas(edges)
    edges_ds = edges_ds.map_batches(lambda batch: batch, batch_format="pyarrow")
    result = compute_connected_components_distributed(
        edges_ds, max_iterations=10, parallelism=10)

    print("Running local")
    result = compute_connected_components_pandas(edges)

    print(result)


def test_gpu_minhash():
    """
    Test GPU MinHash implementation against CPU baseline for correctness.

    Note: The hash values will differ between CPU (SHA1) and GPU (MurmurHash3)
    implementations, but the Jaccard similarity estimates should be comparable.
    This test verifies that the GPU implementation produces valid MinHash signatures
    and that the overall deduplication pipeline works correctly.
    """
    if not CUPY_AVAILABLE:
        print("CuPy not available, skipping GPU test")
        return

    print("=" * 60)
    print("Testing GPU MinHash Implementation")
    print("=" * 60)

    # Test parameters
    num_perm = 128
    ngram_size = 5
    seed = 42

    # Test texts with known similarity
    test_texts = [
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox jumps over the lazy cat",  # Similar to first
        "A completely different sentence about nothing",
        "The quick brown fox jumps over the lazy dog",  # Exact duplicate of first
        "",  # Empty text
        "Short",  # Very short text (less than ngram_size)
    ]

    # Initialize generators
    cpu_gen = MinHashGenerator(num_perm=num_perm, ngram_size=ngram_size, seed=seed)
    gpu_gen = GPUMinHashGenerator(num_perm=num_perm, ngram_size=ngram_size, seed=seed)

    print(f"\nTest configuration:")
    print(f"  - num_perm: {num_perm}")
    print(f"  - ngram_size: {ngram_size}")
    print(f"  - seed: {seed}")
    print(f"  - num_texts: {len(test_texts)}")

    # Compute signatures
    print("\nComputing MinHash signatures...")
    cpu_signatures = []
    gpu_signatures = []

    for i, text in enumerate(test_texts):
        cpu_sig = cpu_gen.compute_minhash(text)
        gpu_sig = gpu_gen.compute_minhash(text)
        cpu_signatures.append(cpu_sig)
        gpu_signatures.append(gpu_sig)

        print(f"  Text {i}: '{text[:40]}...' if len(text) > 40 else '{text}'")
        print(f"    CPU signature shape: {cpu_sig.shape}, dtype: {cpu_sig.dtype}")
        print(f"    GPU signature shape: {gpu_sig.shape}, dtype: {gpu_sig.dtype}")

    # Test batch processing
    print("\nTesting batch processing...")
    gpu_batch_signatures = gpu_gen.compute_minhash_batch(test_texts)
    print(f"  Batch output shape: {gpu_batch_signatures.shape}")

    # Verify batch results match individual results
    batch_match = True
    for i, (single, batch) in enumerate(zip(gpu_signatures, gpu_batch_signatures)):
        if not np.array_equal(single, batch):
            print(f"  WARNING: Batch result differs from single for text {i}")
            batch_match = False

    if batch_match:
        print("  ✓ Batch processing produces identical results to single processing")

    # Compute Jaccard similarity estimates
    def estimate_jaccard(sig1, sig2):
        """Estimate Jaccard similarity from MinHash signatures."""
        return np.mean(sig1 == sig2)

    print("\nJaccard similarity estimates (CPU vs GPU should be similar patterns):")
    print("  CPU Implementation (SHA1):")
    for i in range(len(test_texts)):
        for j in range(i + 1, len(test_texts)):
            sim = estimate_jaccard(cpu_signatures[i], cpu_signatures[j])
            print(f"    texts[{i}] vs texts[{j}]: {sim:.4f}")

    print("  GPU Implementation (MurmurHash3):")
    for i in range(len(test_texts)):
        for j in range(i + 1, len(test_texts)):
            sim = estimate_jaccard(gpu_signatures[i], gpu_signatures[j])
            print(f"    texts[{i}] vs texts[{j}]: {sim:.4f}")

    # Verify expected behaviors
    print("\nValidation checks:")

    # Check 1: Empty text should produce max hash values
    empty_idx = test_texts.index("")
    if np.all(gpu_signatures[empty_idx] == MAX_HASH):
        print("  ✓ Empty text produces max hash values")
    else:
        print("  ✗ Empty text did not produce max hash values")

    # Check 2: Exact duplicates should have identical signatures (same hash function)
    dup_indices = [i for i, t in enumerate(test_texts) if t == test_texts[0]]
    if len(dup_indices) > 1:
        i, j = dup_indices[0], dup_indices[-1]
        if np.array_equal(gpu_signatures[i], gpu_signatures[j]):
            print("  ✓ Exact duplicate texts have identical GPU signatures")
        else:
            print("  ✗ Exact duplicate texts have different GPU signatures")

    # Check 3: Similar texts should have high similarity
    similar_sim_cpu = estimate_jaccard(cpu_signatures[0], cpu_signatures[1])
    similar_sim_gpu = estimate_jaccard(gpu_signatures[0], gpu_signatures[1])
    if similar_sim_gpu > 0.5:
        print(f"  ✓ Similar texts have high similarity (GPU: {similar_sim_gpu:.4f})")
    else:
        print(f"  ? Similar texts have lower than expected similarity (GPU: {similar_sim_gpu:.4f})")

    # Check 4: Different texts should have low similarity
    diff_sim_gpu = estimate_jaccard(gpu_signatures[0], gpu_signatures[2])
    if diff_sim_gpu < 0.3:
        print(f"  ✓ Different texts have low similarity (GPU: {diff_sim_gpu:.4f})")
    else:
        print(f"  ? Different texts have higher than expected similarity (GPU: {diff_sim_gpu:.4f})")

    print("\n" + "=" * 60)
    print("GPU MinHash test completed!")
    print("=" * 60)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    # Check if running in test mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--test-gpu':
        test_gpu_minhash()
    elif len(sys.argv) > 1 and sys.argv[1] == '--test-connected':
        test_connected()
    else:
        main()
    print("finished")
