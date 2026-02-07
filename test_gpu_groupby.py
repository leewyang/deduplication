"""
Unit tests for the GPUDataset groupby().map_groups() API.

Tests cover:
- Basic groupby + map_groups (single key, block and group fn_type)
- Multi-key groupby
- Identity/passthrough, aggregation, and custom UDFs
- Edge cases (empty dataset, single row, single group)
- API compatibility (kwargs ignored, materialize)
- Correctness vs expected results
"""

import logging

import cudf
import pandas as pd
import pytest
import ray

import gpu_dataset  # noqa: F401 - install GPU/CPU methods
from gpu_dataset import GPUDataset


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def ray_context():
    """Initialize Ray for all tests."""
    if not ray.is_initialized():
        ray.init(num_gpus=1, num_cpus=4)
    yield
    ray.shutdown()


# --- UDFs for map_groups (block: receive full partition; group: receive single group) ---


def identity_block(cdf: cudf.DataFrame) -> cudf.DataFrame:
    """Passthrough: return partition as-is."""
    return cdf


def sum_per_key_block(cdf: cudf.DataFrame) -> cudf.DataFrame:
    """Aggregate: one row per key with sum of 'value'."""
    if len(cdf) == 0:
        return cudf.DataFrame({"key": [], "value": []})
    return cudf.DataFrame(
        {"key": cdf["key"].unique(), "value": cdf.groupby("key")["value"].sum()}
    ).reset_index(drop=True)


def sum_per_key_block_typed(cdf: cudf.DataFrame) -> cudf.DataFrame:
    """Aggregate: key and sum(value) per key, explicit columns."""
    if len(cdf) == 0:
        return cudf.DataFrame({"key": cudf.Series(dtype="int64"), "total": cudf.Series(dtype="int64")})
    out = cdf.groupby("key", as_index=False).agg({"value": "sum"})
    out = out.rename(columns={"value": "total"})
    return out


def first_per_key_group(cdf: cudf.DataFrame) -> cudf.DataFrame:
    """Group UDF: return first row of each group (group has one key value)."""
    return cdf.head(1)


def count_per_key_block(cdf: cudf.DataFrame) -> cudf.DataFrame:
    """One row per key with count."""
    if len(cdf) == 0:
        return cudf.DataFrame({"key": [], "count": []})
    cnt = cdf.groupby("key").size().reset_index(name="count")
    return cnt


def multi_key_agg_block(cdf: cudf.DataFrame) -> cudf.DataFrame:
    """Aggregate by (a, b): sum of v."""
    if len(cdf) == 0:
        return cudf.DataFrame({"a": [], "b": [], "v": []})
    return cdf.groupby(["a", "b"], as_index=False)["v"].sum()


# ---------------------------------------------------------------------------


class TestGroupbyMapGroupsBasic:
    """Basic groupby().map_groups() behavior."""

    def test_groupby_single_key_map_groups_identity_block(self, ray_context):
        """groupby one key + map_groups identity (block) preserves rows."""
        ds = ray.data.from_items(
            [{"key": i % 3, "value": i * 10} for i in range(30)]
        )
        result = (
            ds.gpu(nranks=1)
            .groupby("key", num_partitions=2)
            .map_groups(identity_block, fn_type="block")
            .materialize()
        )
        result_df = result.to_pandas().sort_values(["key", "value"]).reset_index(drop=True)
        expected_df = pd.DataFrame(
            [{"key": i % 3, "value": i * 10} for i in range(30)]
        ).sort_values(["key", "value"]).reset_index(drop=True)
        assert result.count() == 30
        pd.testing.assert_frame_equal(result_df, expected_df)
        logger.info("✓ test_groupby_single_key_map_groups_identity_block passed")

    def test_groupby_single_key_map_groups_aggregation_block(self, ray_context):
        """groupby one key + map_groups sum per key (block)."""
        ds = ray.data.from_items(
            [{"key": 0, "value": 1}, {"key": 0, "value": 2}, {"key": 1, "value": 10}]
        )
        result = (
            ds.gpu(nranks=1)
            .groupby("key", num_partitions=2)
            .map_groups(sum_per_key_block_typed, fn_type="block")
            .materialize()
        )
        result_df = result.to_pandas().sort_values("key").reset_index(drop=True)
        expected_df = pd.DataFrame({"key": [0, 1], "total": [3, 10]})
        assert result.count() == 2
        pd.testing.assert_frame_equal(result_df, expected_df)
        logger.info("✓ test_groupby_single_key_map_groups_aggregation_block passed")

    def test_groupby_single_key_map_groups_group_fn_type(self, ray_context):
        """groupby + map_groups with fn_type='group' (one group at a time)."""
        ds = ray.data.from_items(
            [{"key": 0, "value": 1}, {"key": 0, "value": 2}, {"key": 1, "value": 10}]
        )
        print(ds)
        result = (
            ds.gpu(nranks=1)
            .groupby("key", num_partitions=2)
            .map_groups(first_per_key_group, fn_type="group")
            .materialize()
        )
        result_df = result.to_pandas().sort_values("key").reset_index(drop=True)
        # One row per key (first row of each group)
        assert result.count() == 2
        assert set(result_df["key"].tolist()) == {0, 1}
        assert set(result_df["value"].tolist()) == {1, 10}
        logger.info("✓ test_groupby_single_key_map_groups_group_fn_type passed")

    def test_groupby_returns_gpudataset_chainable(self, ray_context):
        """groupby() returns self; map_groups() returns self; can chain materialize()."""
        ds = ray.data.from_items([{"key": 1, "value": 2}])
        gpu = ds.gpu(nranks=1)
        after_groupby = gpu.groupby("key", num_partitions=1)
        assert after_groupby is gpu
        after_map = after_groupby.map_groups(identity_block, fn_type="block")
        assert after_map is gpu
        out = after_map.materialize()
        assert out.count() == 1
        logger.info("✓ test_groupby_returns_gpudataset_chainable passed")

    def test_map_groups_accepts_kwargs(self, ray_context):
        """map_groups() accepts **kwargs for API compatibility (e.g. batch_format)."""
        ds = ray.data.from_items([{"key": 1, "value": 2}])
        result = (
            ds.gpu(nranks=1)
            .groupby("key", num_partitions=1)
            .map_groups(identity_block, fn_type="block", batch_format="pyarrow")
            .materialize()
        )
        assert result.count() == 1
        logger.info("✓ test_map_groups_accepts_kwargs passed")


class TestGroupbyMultiKey:
    """Multi-key groupby."""

    def test_groupby_multi_key_map_groups_block(self, ray_context):
        """groupby([a, b]) + map_groups aggregation."""
        ds = ray.data.from_items(
            [
                {"a": 1, "b": "x", "v": 10},
                {"a": 1, "b": "x", "v": 20},
                {"a": 1, "b": "y", "v": 5},
                {"a": 2, "b": "x", "v": 1},
            ]
        )
        result = (
            ds.gpu(nranks=1)
            .groupby(["a", "b"], num_partitions=2)
            .map_groups(multi_key_agg_block, fn_type="block")
            .materialize()
        )
        result_df = result.to_pandas().sort_values(["a", "b"]).reset_index(drop=True)
        expected_df = pd.DataFrame(
            [
                {"a": 1, "b": "x", "v": 30},
                {"a": 1, "b": "y", "v": 5},
                {"a": 2, "b": "x", "v": 1},
            ]
        )
        assert result.count() == 3
        pd.testing.assert_frame_equal(result_df, expected_df)
        logger.info("✓ test_groupby_multi_key_map_groups_block passed")


class TestGroupbyEdgeCases:
    """Edge cases: empty, single row, single group."""

    def test_groupby_empty_dataset(self, ray_context):
        """Empty dataset: groupby + map_groups returns empty result."""
        ds = ray.data.from_items([])
        # Empty dataset may not have columns; use from_pandas to get schema
        empty_df = pd.DataFrame(columns=["key", "value"])
        ds = ray.data.from_pandas(empty_df)
        result = (
            ds.gpu(nranks=1)
            .groupby("key", num_partitions=1)
            .map_groups(identity_block, fn_type="block")
            .materialize()
        )
        assert result.count() == 0
        logger.info("✓ test_groupby_empty_dataset passed")

    def test_groupby_single_row(self, ray_context):
        """Single row: one group, map_groups returns one row."""
        ds = ray.data.from_items([{"key": 42, "value": 100}])
        result = (
            ds.gpu(nranks=1)
            .groupby("key", num_partitions=1)
            .map_groups(identity_block, fn_type="block")
            .materialize()
        )
        assert result.count() == 1
        row = result.to_pandas().iloc[0]
        assert row["key"] == 42 and row["value"] == 100
        logger.info("✓ test_groupby_single_row passed")

    def test_groupby_single_group_multiple_rows(self, ray_context):
        """All rows same key: one group, aggregation yields one row."""
        ds = ray.data.from_items([{"key": 0, "value": i} for i in range(5)])
        result = (
            ds.gpu(nranks=1)
            .groupby("key", num_partitions=1)
            .map_groups(sum_per_key_block_typed, fn_type="block")
            .materialize()
        )
        assert result.count() == 1
        row = result.to_pandas().iloc[0]
        assert row["key"] == 0 and row["total"] == 0 + 1 + 2 + 3 + 4
        logger.info("✓ test_groupby_single_group_multiple_rows passed")


class TestGroupbyCorrectness:
    """Correctness: compare to pandas/cudf expected results."""

    def test_count_per_key_matches_expected(self, ray_context):
        """count per key matches manual expectation."""
        items = [{"key": 0, "value": 1}, {"key": 0, "value": 2}, {"key": 1, "value": 3}]
        ds = ray.data.from_items(items)
        result = (
            ds.gpu(nranks=1)
            .groupby("key", num_partitions=2)
            .map_groups(count_per_key_block, fn_type="block")
            .materialize()
        )
        result_df = result.to_pandas().sort_values("key").reset_index(drop=True)
        expected = pd.DataFrame({"key": [0, 1], "count": [2, 1]})
        pd.testing.assert_frame_equal(result_df, expected)
        logger.info("✓ test_count_per_key_matches_expected passed")

    def test_larger_dataset_row_count_preserved_identity(self, ray_context):
        """Larger dataset: identity map_groups preserves total row count."""
        n = 200
        ds = ray.data.from_items([{"key": i % 10, "value": i} for i in range(n)])
        result = (
            ds.gpu(nranks=1)
            .groupby("key", num_partitions=4)
            .map_groups(identity_block, fn_type="block")
            .materialize()
        )
        assert result.count() == n
        result_df = result.to_pandas()
        assert result_df["key"].nunique() == 10
        assert result_df["value"].sum() == sum(range(n))
        logger.info("✓ test_larger_dataset_row_count_preserved_identity passed")


class TestGroupbyAPI:
    """API surface: num_partitions, key types."""

    def test_groupby_default_num_partitions(self, ray_context):
        """groupby(key) without num_partitions uses nranks."""
        ds = ray.data.from_items([{"key": 1, "value": 2}])
        result = (
            ds.gpu(nranks=1)
            .groupby("key")
            .map_groups(identity_block, fn_type="block")
            .materialize()
        )
        assert result.count() == 1
        logger.info("✓ test_groupby_default_num_partitions passed")

    def test_groupby_key_as_list_single(self, ray_context):
        """groupby([key]) with list of one key works."""
        ds = ray.data.from_items([{"key": 1, "value": 2}])
        result = (
            ds.gpu(nranks=1)
            .groupby(["key"], num_partitions=1)
            .map_groups(identity_block, fn_type="block")
            .materialize()
        )
        assert result.count() == 1
        logger.info("✓ test_groupby_key_as_list_single passed")

    def test_materialize_returns_ray_dataset(self, ray_context):
        """materialize() returns ray.data.Dataset (not GPUDataset)."""
        ds = ray.data.from_items([{"key": 1, "value": 2}])
        result = (
            ds.gpu(nranks=1)
            .groupby("key", num_partitions=1)
            .map_groups(identity_block, fn_type="block")
            .materialize()
        )
        assert isinstance(result, ray.data.Dataset)
        assert not isinstance(result, GPUDataset)
        logger.info("✓ test_materialize_returns_ray_dataset passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
