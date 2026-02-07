import logging
from typing import Any, Callable, Iterator, Literal, Optional

import cudf
import pyarrow as pa
import ray
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
    ) -> Iterator[pa.Table]:
        """Extract partitions in a streaming manner, yielding each as soon as it's ready.

        Since data is hash-partitioned by shuffle keys, each partition contains complete
        groups, allowing us to apply the function to each partition independently.

        This is a generator method - when called via Ray remote, it returns an
        ObjectRefGenerator that yields partitions as they become available.

        Args:
            map_groups_fn: Optional function to apply to each partition.
            fn_type: "block" applies fn to entire partition, "group" applies fn per group.

        Yields:
            pa.Table: Each partition as a PyArrow Table.
        """
        for partition_idx, (_, partition) in enumerate(self.extract()):
            # Convert partition to cuDF DataFrame immediately. Copy to materialize
            # spillable buffers and avoid "An owning spillable buffer must either be
            # exposed or spill locked" when groupby().apply() concats chunk results.
            cdf = pylibcudf_to_cudf_dataframe(partition, self.columns).copy(deep=True)

            # Apply the map_groups_fn to this partition, if provided
            if map_groups_fn:
                if fn_type == "group":
                    # result = cdf.groupby(self.group_by).apply(map_groups_fn)
                    # Manual group loop + concat with deep copy to avoid spillable buffer
                    # issues in cuDF's internal concat of apply() results.
                    groups = [grp for _, grp in cdf.groupby(self.group_by)]
                    if groups:
                        chunk_results = [map_groups_fn(grp) for grp in groups]
                        result = cudf.concat(
                            [r.copy(deep=True) for r in chunk_results],
                            ignore_index=True,
                        )
                    else:
                        result = cudf.DataFrame({col: [] for col in cdf.columns})
                elif fn_type == "block":
                    result = map_groups_fn(cdf)
                else:
                    raise ValueError(f"Invalid fn_type: {fn_type}")
            else:
                result = cdf if len(cdf) > 0 else cudf.DataFrame({col: [] for col in self.columns})

            logger.debug(f"Rank {self.rank}: Yielding partition {partition_idx} with {len(result)} rows")
            # preserve_index=False avoids "Cannot insert 'key', already exists" when groupby().apply()
            # returns a result with the group key in the index and also in the columns
            yield result.to_arrow(preserve_index=False)