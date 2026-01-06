# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from: https://github.com/NVIDIA-NeMo/Curator/blob/030ce4f597fefb96ba78ca24d5360379ef0ccb54/nemo_curator/stages/deduplication/fuzzy/minhash.py

from abc import ABC
import numpy as np
import cudf
import rmm

import logging

logging.basicConfig(
    format='%(asctime)s  %(levelname)s %(name)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class GPUMinHash(ABC):
    def __init__(
        self,
        seed: int = 42,
        num_hashes: int = 260,
        char_ngrams: int = 24,
        use_64bit_hash: bool = False,
        pool: bool = False,
    ):
        self.seed = seed
        self.num_hashes = num_hashes
        self.char_ngrams = char_ngrams
        self.use_64bit_hash = use_64bit_hash

        # Initialize memory pool for cuDF
        if pool:
            rmm.reinitialize(pool_allocator=pool)

        # Generate seeds
        self.seeds = self.generate_seeds(
            n_permutations=self.num_hashes,
            seed=self.seed,
            bit_width=64 if self.use_64bit_hash else 32,
        )

    def generate_seeds(self, n_permutations: int = 260, seed: int = 0, bit_width: int = 32) -> np.ndarray:
        """
        Generate seeds for all minhash permutations based on the given seed.
        """
        gen = np.random.RandomState(seed)

        if bit_width == 32:  # noqa: PLR2004
            MERSENNE_PRIME = np.uint32((1 << 31) - 1)  # noqa: N806
            dtype = np.uint32
        elif bit_width == 64:  # noqa: PLR2004
            # For 64-bit, use a larger prime number suitable for 64-bit operations
            MERSENNE_PRIME = np.uint64((1 << 61) - 1)  # noqa: N806
            dtype = np.uint64
        else:
            msg = "Unsupported bit width. Use either 32 or 64."
            raise ValueError(msg)

        return np.array(
            [
                (
                    gen.randint(1, MERSENNE_PRIME, dtype=dtype),
                    gen.randint(0, MERSENNE_PRIME, dtype=dtype),
                )
                for _ in range(n_permutations)
            ],
            dtype=dtype,
        )

    def minhash32(self, ser: cudf.Series) -> cudf.Series:
        """
        Compute 32bit minhashes based on the MurmurHash3 algorithm
        """
        if not isinstance(ser, cudf.Series):
            msg = "Expected data of type cudf.Series"
            raise TypeError(msg)

        seeds_a = cudf.Series(self.seeds[:, 0], dtype="uint32")
        seeds_b = cudf.Series(self.seeds[:, 1], dtype="uint32")

        return ser.str.minhash(a=seeds_a, b=seeds_b, seed=self.seeds[0][0], width=self.char_ngrams)

    def minhash64(self, ser: cudf.Series) -> cudf.Series:
        """
        Compute 64bit minhashes based on the MurmurHash3 algorithm
        """
        if not isinstance(ser, cudf.Series):
            msg = "Expected data of type cudf.Series"
            raise TypeError(msg)

        seeds_a = cudf.Series(self.seeds[:, 0], dtype="uint64")
        seeds_b = cudf.Series(self.seeds[:, 1], dtype="uint64")

        return ser.str.minhash64(a=seeds_a, b=seeds_b, seed=self.seeds[0][0], width=self.char_ngrams)

    def compute_minhashes(self, text_series: cudf.Series) -> cudf.Series:
        """
        Compute minhash signatures for the given text series.

        Parameters
        ----------
        text_series: cudf.Series
            Series containing text data to compute minhashes for

        Returns
        -------
        cudf.Series containing minhash signatures
        """
        if not isinstance(text_series, cudf.Series):
            msg = "Expected data of type cudf.Series"
            raise TypeError(msg)

        # Compute minhashes
        minhash_method = self.minhash64 if self.use_64bit_hash else self.minhash32
        return minhash_method(text_series)


if __name__ == "__main__":
    import glob
    import pandas as pd
    import ray
    import time

    input_files = sorted(glob.glob("/raid/spark-team/leey/ray-data/fineweb-edu-10/*.parquet"))
    generator = GPUMinHash(seed=42, num_hashes=260, char_ngrams=24, use_64bit_hash=False, pool=True)

    # cuDF: everything on GPU, results in list
    minhashes_list = []
    start = time.time()
    for input_file in input_files:
        df = cudf.read_parquet(input_file)
        text_series = cudf.Series(df["text"])
        minhashes = generator.compute_minhashes(text_series)
        minhashes_list.append(minhashes)
    stop = time.time()
    cudf_list_time = stop - start
    logger.debug(minhashes_list)
    logger.info(f"===== cuDF (list): {cudf_list_time} seconds")

    # cuDF: read on CPU, compute on GPU, results in list
    minhashes_list = []
    start = time.time()
    for input_file in input_files:
        df = pd.read_parquet(input_file)
        text_series = cudf.Series(df["text"])
        minhashes = generator.compute_minhashes(text_series)
        minhashes_list.append(minhashes)
    stop = time.time()
    cudf_list_time = stop - start
    logger.debug(minhashes_list)
    logger.info(f"===== cuDF (list + CPU): {cudf_list_time} seconds")

    # cuDF: everything on GPU, results in concatenated cudf Series
    minhashes_list = []
    start = time.time()
    for input_file in input_files:
        df = cudf.read_parquet(input_file)
        text_series = cudf.Series(df["text"])
        minhashes = generator.compute_minhashes(text_series)
        minhashes_list.append(minhashes)
    df = cudf.concat(minhashes_list)
    stop = time.time()
    cudf_concat_time = stop - start
    logger.debug(df)
    logger.info(f"===== cuDF (concat): {cudf_concat_time} seconds")

    # cuDF: everything on GPU, results in concatenated cudf Series, then to pandas
    minhashes_list = []
    start = time.time()
    for input_file in input_files:
        df = cudf.read_parquet(input_file)
        text_series = cudf.Series(df["text"])
        minhashes = generator.compute_minhashes(text_series)
        minhashes_list.append(minhashes)
    df = cudf.concat(minhashes_list)
    stop = time.time()
    logger.info(f"===== cuDF (concat + pandas): concat: {stop - start} seconds")
    start_to_pandas = time.time()
    pdf = df.to_pandas()
    stop_to_pandas = time.time()
    logger.info(f"===== cuDF (concat + pandas): to_pandas: {stop_to_pandas - start_to_pandas} seconds")
    cudf_pandas_time = stop_to_pandas - start
    logger.debug(pdf)
    logger.info(f"===== cuDF (concat + pandas): {cudf_pandas_time} total seconds")

    # cuDF: everything on GPU, results in concatenated cudf Series, then to numpy
    minhashes_list = []
    start = time.time()
    for input_file in input_files:
        df = cudf.read_parquet(input_file)
        text_series = cudf.Series(df["text"])
        minhashes = generator.compute_minhashes(text_series)
        minhashes_list.append(minhashes)
    df = cudf.concat(minhashes_list)
    stop = time.time()
    logger.info(f"===== cuDF (concat + numpy): concat: {stop - start} seconds")
    start_to_numpy = time.time()
    flat_values = df.list.leaves.values
    numpy_array = flat_values.get().reshape(-1, 260)
    stop_to_numpy = time.time()
    logger.info(f"===== cuDF (concat + numpy): to_numpy: {stop_to_numpy - start_to_numpy} seconds")
    cudf_numpy_time = stop_to_numpy - start
    logger.debug(pdf)
    logger.info(f"===== cuDF (concat + numpy): {cudf_numpy_time} seconds")

    # cuDF: everything on GPU, results in list of pandas Series, then to pandas concat
    minhashes_list = []
    start = time.time()
    for input_file in input_files:
        df = cudf.read_parquet(input_file)
        text_series = cudf.Series(df["text"])
        minhashes = generator.compute_minhashes(text_series)
        minhashes_list.append(minhashes.to_pandas())
    df = pd.concat(minhashes_list)
    stop = time.time()
    cudf_list_pandas_time = stop - start
    logger.debug(df)
    logger.info(f"===== cuDF (list + pandas): {cudf_list_pandas_time} seconds")

    # Ray implementation
    ray.init()

    # disable progress bars
    ray.data.DataContext.get_current().enable_progress_bars = False

    # Ray: minhash on GPU, input and output on CPU, pandas batch format
    start = time.time()
    def minhash_gpu(x):
        text_series = cudf.Series(x["text"])
        minhashes = generator.compute_minhashes(text_series).to_pandas()
        return pd.DataFrame({"minhashes": minhashes})
    ds = ray.data.read_parquet(input_files)
    minhashes = ds.map_batches(minhash_gpu, batch_format='pandas', batch_size=1000*10, num_gpus=1)
    minhashes = minhashes.to_pandas()
    stop = time.time()
    ray_time = stop - start
    logger.debug(minhashes)
    logger.info(f"===== Ray (pandas): {ray_time} seconds")

    # Ray: minhash on GPU, input and output on CPU, numpy batch format
    start = time.time()
    def minhash_gpu(x):
        text_series = cudf.Series(x["text"])
        minhashes = generator.compute_minhashes(text_series).list.leaves.values.get().reshape(-1, 260)
        return {"minhashes": minhashes}
    ds = ray.data.read_parquet(input_files)
    minhashes = ds.map_batches(minhash_gpu, batch_format='numpy', batch_size=1000*10, num_gpus=1)
    minhashes = minhashes.to_pandas()
    stop = time.time()
    ray_numpy_time = stop - start
    logger.debug(minhashes)
    logger.info(f"===== Ray (numpy): {ray_numpy_time} seconds")
