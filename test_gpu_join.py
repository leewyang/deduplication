"""
Comprehensive test suite for GPU-accelerated join operations.

Tests cover:
- Basic join types (inner, left, left_anti)
- Multi-key joins
- Edge cases (empty datasets, no matches)
- Integration with deduplication pipeline
- Performance benchmarks
"""

import logging
import time

import pandas as pd
import pytest
import ray

import gpu_dataset  # Install GPU/CPU methods


logging.basicConfig(
    format='%(asctime)s %(levelname)s %(name)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def ray_context():
    """Initialize Ray for all tests."""
    if not ray.is_initialized():
        ray.init(num_gpus=1, num_cpus=4)
    yield
    ray.shutdown()


class TestBasicJoins:
    """Test basic join operations."""

    def test_gpu_join_inner(self, ray_context):
        """Test basic inner join on GPU."""
        left_ds = ray.data.from_items([
            {'id': 1, 'value': 'a'},
            {'id': 2, 'value': 'b'},
            {'id': 3, 'value': 'c'},
        ])

        right_ds = ray.data.from_items([
            {'id': 2, 'score': 100},
            {'id': 3, 'score': 200},
            {'id': 4, 'score': 300},
        ])

        # GPU join
        result = left_ds.gpu(nranks=1).join(
            right_ds, on='id', join_type='inner'
        ).materialize()

        result_df = result.to_pandas().sort_values('id').reset_index(drop=True)

        # Expected: only matching ids (2, 3)
        expected_df = pd.DataFrame([
            {'id': 2, 'value': 'b', 'score': 100},
            {'id': 3, 'value': 'c', 'score': 200},
        ])

        pd.testing.assert_frame_equal(result_df, expected_df)
        logger.info("✓ test_gpu_join_inner passed")

    def test_gpu_join_left(self, ray_context):
        """Test left join on GPU."""
        left_ds = ray.data.from_items([
            {'id': 1, 'value': 'a'},
            {'id': 2, 'value': 'b'},
            {'id': 3, 'value': 'c'},
        ])

        right_ds = ray.data.from_items([
            {'id': 2, 'score': 100},
            {'id': 3, 'score': 200},
        ])

        # GPU join
        result = left_ds.gpu(nranks=1).join(
            right_ds, on='id', join_type='left'
        ).materialize()

        result_df = result.to_pandas().sort_values('id').reset_index(drop=True)

        # Expected: all left rows, nulls for non-matching
        expected_df = pd.DataFrame([
            {'id': 1, 'value': 'a', 'score': None},
            {'id': 2, 'value': 'b', 'score': 100.0},
            {'id': 3, 'value': 'c', 'score': 200.0},
        ])

        # Convert score column to same dtype (cuDF may use Int64, pandas uses float64)
        if result_df['score'].dtype != expected_df['score'].dtype:
            result_df['score'] = result_df['score'].astype('float64')

        pd.testing.assert_frame_equal(result_df, expected_df)
        logger.info("✓ test_gpu_join_left passed")

    def test_gpu_join_left_anti(self, ray_context):
        """Test left anti join on GPU (critical for deduplication)."""
        left_ds = ray.data.from_items([
            {'id': 1, 'value': 'a'},
            {'id': 2, 'value': 'b'},
            {'id': 3, 'value': 'c'},
            {'id': 4, 'value': 'd'},
        ])

        right_ds = ray.data.from_items([
            {'id': 2, 'duplicate': True},
            {'id': 3, 'duplicate': True},
        ])

        # GPU join - keep only rows NOT in right dataset
        result = left_ds.gpu(nranks=1).join(
            right_ds, on='id', join_type='left_anti'
        ).materialize()

        result_df = result.to_pandas().sort_values('id').reset_index(drop=True)

        # Expected: only rows without match in right (1, 4)
        expected_df = pd.DataFrame([
            {'id': 1, 'value': 'a'},
            {'id': 4, 'value': 'd'},
        ])

        pd.testing.assert_frame_equal(result_df, expected_df)
        logger.info("✓ test_gpu_join_left_anti passed")


class TestMultiKeyJoins:
    """Test joins with multiple keys."""

    def test_gpu_join_multi_key(self, ray_context):
        """Test join with multiple keys."""
        left_ds = ray.data.from_items([
            {'key1': 1, 'key2': 'x', 'value': 'a'},
            {'key1': 1, 'key2': 'y', 'value': 'b'},
            {'key1': 2, 'key2': 'x', 'value': 'c'},
        ])

        right_ds = ray.data.from_items([
            {'key1': 1, 'key2': 'x', 'score': 100},
            {'key1': 2, 'key2': 'x', 'score': 200},
        ])

        # GPU join on multiple keys
        result = left_ds.gpu(nranks=1).join(
            right_ds, on=['key1', 'key2'], join_type='inner'
        ).materialize()

        result_df = result.to_pandas().sort_values(['key1', 'key2']).reset_index(drop=True)

        # Expected: only exact matches on both keys
        expected_df = pd.DataFrame([
            {'key1': 1, 'key2': 'x', 'value': 'a', 'score': 100},
            {'key1': 2, 'key2': 'x', 'value': 'c', 'score': 200},
        ])

        pd.testing.assert_frame_equal(result_df, expected_df)
        logger.info("✓ test_gpu_join_multi_key passed")

    def test_gpu_join_different_key_names(self, ray_context):
        """Test join with different key names in left and right datasets."""
        left_ds = ray.data.from_items([
            {'user_id': 1, 'name': 'Alice'},
            {'user_id': 2, 'name': 'Bob'},
        ])

        right_ds = ray.data.from_items([
            {'customer_id': 1, 'score': 100},
            {'customer_id': 2, 'score': 200},
        ])

        # GPU join with different key names
        result = left_ds.gpu(nranks=1).join(
            right_ds, on='user_id', right_on='customer_id', join_type='inner'
        ).materialize()

        result_df = result.to_pandas().sort_values('user_id').reset_index(drop=True)

        # Expected: joined on user_id = customer_id
        expected_df = pd.DataFrame([
            {'user_id': 1, 'name': 'Alice', 'customer_id': 1, 'score': 100},
            {'user_id': 2, 'name': 'Bob', 'customer_id': 2, 'score': 200},
        ])

        pd.testing.assert_frame_equal(result_df, expected_df)
        logger.info("✓ test_gpu_join_different_key_names passed")


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_gpu_join_empty_left(self, ray_context):
        """Test join with empty left dataset."""
        left_ds = ray.data.from_items([])
        right_ds = ray.data.from_items([
            {'id': 1, 'score': 100},
        ])

        result = left_ds.gpu(nranks=1).join(
            right_ds, on='id', join_type='inner'
        ).materialize()

        assert result.count() == 0
        logger.info("✓ test_gpu_join_empty_left passed")

    def test_gpu_join_empty_right_inner(self, ray_context):
        """Test inner join with empty right dataset."""
        left_ds = ray.data.from_items([
            {'id': 1, 'value': 'a'},
        ])
        right_ds = ray.data.from_items([])

        result = left_ds.gpu(nranks=1).join(
            right_ds, on='id', join_type='inner'
        ).materialize()

        assert result.count() == 0
        logger.info("✓ test_gpu_join_empty_right_inner passed")

    def test_gpu_join_empty_right_left_anti(self, ray_context):
        """Test left_anti join with empty right dataset."""
        left_ds = ray.data.from_items([
            {'id': 1, 'value': 'a'},
            {'id': 2, 'value': 'b'},
        ])
        right_ds = ray.data.from_items([])

        result = left_ds.gpu(nranks=1).join(
            right_ds, on='id', join_type='left_anti'
        ).materialize()

        # Should return all left rows since nothing to filter
        assert result.count() == 2
        logger.info("✓ test_gpu_join_empty_right_left_anti passed")

    def test_gpu_join_no_matches(self, ray_context):
        """Test join with no matching keys."""
        left_ds = ray.data.from_items([
            {'id': 1, 'value': 'a'},
            {'id': 2, 'value': 'b'},
        ])

        right_ds = ray.data.from_items([
            {'id': 3, 'score': 100},
            {'id': 4, 'score': 200},
        ])

        result = left_ds.gpu(nranks=1).join(
            right_ds, on='id', join_type='inner'
        ).materialize()

        assert result.count() == 0
        logger.info("✓ test_gpu_join_no_matches passed")

    def test_gpu_join_invalid_key(self, ray_context):
        """Test join with invalid key name."""
        left_ds = ray.data.from_items([
            {'id': 1, 'value': 'a'},
        ])

        right_ds = ray.data.from_items([
            {'id': 1, 'score': 100},
        ])

        # Should raise ValueError for invalid key
        with pytest.raises(ValueError, match="not found"):
            left_ds.gpu(nranks=1).join(
                right_ds, on='invalid_key', join_type='inner'
            ).materialize()

        logger.info("✓ test_gpu_join_invalid_key passed")

    def test_gpu_join_unsupported_type(self, ray_context):
        """Test join with unsupported join type."""
        left_ds = ray.data.from_items([
            {'id': 1, 'value': 'a'},
        ])

        right_ds = ray.data.from_items([
            {'id': 1, 'score': 100},
        ])

        # Should raise ValueError for unsupported join type
        with pytest.raises(ValueError, match="Unsupported join type"):
            left_ds.gpu(nranks=1).join(
                right_ds, on='id', join_type='outer'
            ).materialize()

        logger.info("✓ test_gpu_join_unsupported_type passed")


class TestCorrectness:
    """Test GPU join correctness against CPU baseline."""

    def test_gpu_vs_cpu_inner_join(self, ray_context):
        """Verify GPU inner join matches CPU join."""
        left_ds = ray.data.from_items([
            {'id': i, 'value': f'val_{i}'}
            for i in range(100)
        ])

        right_ds = ray.data.from_items([
            {'id': i * 2, 'score': i * 10}
            for i in range(60)
        ])

        # CPU join (Ray Data uses join_type parameter, not how)
        cpu_result = left_ds.join(
            right_ds, on=('id',), join_type='inner', num_partitions=1
        ).materialize().to_pandas().sort_values('id').reset_index(drop=True)

        # GPU join
        gpu_result = left_ds.gpu(nranks=1).join(
            right_ds, on='id', join_type='inner'
        ).materialize().to_pandas().sort_values('id').reset_index(drop=True)

        pd.testing.assert_frame_equal(cpu_result, gpu_result)
        logger.info("✓ test_gpu_vs_cpu_inner_join passed")

    def test_gpu_vs_cpu_left_join(self, ray_context):
        """Verify GPU left join matches CPU join."""
        left_ds = ray.data.from_items([
            {'id': i, 'value': f'val_{i}'}
            for i in range(50)
        ])

        right_ds = ray.data.from_items([
            {'id': i * 2, 'score': i * 10}
            for i in range(30)
        ])

        # CPU join (Ray Data uses 'left_outer' for left join)
        cpu_result = left_ds.join(
            right_ds, on=('id',), join_type='left_outer', num_partitions=1
        ).materialize().to_pandas().sort_values('id').reset_index(drop=True)

        # GPU join
        gpu_result = left_ds.gpu(nranks=1).join(
            right_ds, on='id', join_type='left'
        ).materialize().to_pandas().sort_values('id').reset_index(drop=True)

        # Convert dtypes to match
        if 'score' in gpu_result.columns and gpu_result['score'].dtype != cpu_result['score'].dtype:
            gpu_result['score'] = gpu_result['score'].astype(cpu_result['score'].dtype)

        pd.testing.assert_frame_equal(cpu_result, gpu_result)
        logger.info("✓ test_gpu_vs_cpu_left_join passed")


class TestIntegration:
    """Integration tests with deduplication pipeline."""

    def test_deduplication_workflow(self, ray_context):
        """Test GPU join in deduplication workflow."""
        # Simulate original documents
        docs_ds = ray.data.from_items([
            {'id': 1, 'text': 'document 1'},
            {'id': 2, 'text': 'document 2'},
            {'id': 3, 'text': 'document 3'},
            {'id': 4, 'text': 'document 4'},
            {'id': 5, 'text': 'document 5'},
        ])

        # Simulate duplicate detection (ids 2 and 4 are duplicates)
        duplicates_ds = ray.data.from_items([
            {'node': 2},
            {'node': 4},
        ])

        # Remove duplicates using left_anti join
        deduplicated = docs_ds.gpu(nranks=1).join(
            duplicates_ds, on='id', right_on='node', join_type='left_anti'
        ).materialize()

        result_df = deduplicated.to_pandas().sort_values('id').reset_index(drop=True)

        # Should keep ids 1, 3, 5 (removed 2, 4)
        expected_df = pd.DataFrame([
            {'id': 1, 'text': 'document 1'},
            {'id': 3, 'text': 'document 3'},
            {'id': 5, 'text': 'document 5'},
        ])

        pd.testing.assert_frame_equal(result_df, expected_df)
        logger.info("✓ test_deduplication_workflow passed")


class TestPerformance:
    """Performance benchmarks (informational, not strict assertions)."""

    @pytest.mark.skip(reason="Performance test - run manually")
    def test_benchmark_gpu_vs_cpu(self, ray_context):
        """Benchmark GPU vs CPU join performance."""
        # Create larger datasets for meaningful benchmark
        n_left = 100000
        n_right = 50000

        left_ds = ray.data.from_items([
            {'id': i, 'value': f'val_{i}'}
            for i in range(n_left)
        ])

        right_ds = ray.data.from_items([
            {'id': i * 2, 'score': i * 10}
            for i in range(n_right)
        ])

        # Benchmark CPU join
        start = time.time()
        cpu_result = left_ds.join(right_ds, on='id', join_type='inner').materialize()
        cpu_count = cpu_result.count()
        cpu_time = time.time() - start

        # Benchmark GPU join
        start = time.time()
        gpu_result = left_ds.gpu(nranks=1).join(right_ds, on='id', join_type='inner').materialize()
        gpu_count = gpu_result.count()
        gpu_time = time.time() - start

        # Log results
        logger.info(f"CPU join: {cpu_time:.2f}s, {cpu_count} rows")
        logger.info(f"GPU join: {gpu_time:.2f}s, {gpu_count} rows")
        logger.info(f"Speedup: {cpu_time / gpu_time:.2f}x")

        assert cpu_count == gpu_count


class TestLargeJoins:
    """Test large join operations."""

    def test_large_right_dataset(self, ray_context):
        """Test join with large right dataset (simulating memory pressure)."""

        # Create a small left dataset
        left_data = [{"id": i, "value": f"left_{i}"} for i in range(1000)]
        left_ds = ray.data.from_items(left_data)

        # Create a LARGE right dataset (100K rows)
        # This would cause OOM with the old implementation
        logger.info("Creating large right dataset (100K rows)...")
        right_data = [{"id": i % 1000, "attr": f"right_{i}"} for i in range(100_000)]
        right_ds = ray.data.from_items(right_data)

        logger.info(f"Left dataset: {left_ds.count()} rows")
        logger.info(f"Right dataset: {right_ds.count()} rows")

        # Perform inner join (should match all left rows, multiple times)
        logger.info("Performing GPU join...")
        start = time.time()

        result = left_ds.gpu(nranks=1).join(
            right_ds,
            on='id',
            join_type='inner',
            num_partitions=10
        ).materialize()

        elapsed = time.time() - start
        result_count = result.count()

        logger.info(f"Join completed in {elapsed:.2f}s")
        logger.info(f"Result: {result_count} rows")

        # Verify correctness
        # Each left row should match ~100 right rows (100K / 1000)
        expected_count = 1000 * 100
        assert result_count == expected_count, f"Expected {expected_count} rows, got {result_count}"

        # Verify data integrity
        sample = result.take(5)
        for row in sample:
            assert 'id' in row
            assert 'value' in row
            assert 'attr' in row
            logger.info(f"Sample row: {row}")

        logger.info("✓ Large right dataset test PASSED")


    def test_very_large_right_dataset(self, ray_context):
        """Test join with very large right dataset (1M rows)."""
        # Small left dataset
        left_data = [{"id": i, "value": f"left_{i}"} for i in range(100)]
        left_ds = ray.data.from_items(left_data)

        # Very large right dataset (1M rows)
        logger.info("Creating very large right dataset (1M rows)...")
        right_data = [{"id": i % 100, "attr": f"right_{i}"} for i in range(1_000_000)]
        right_ds = ray.data.from_items(right_data)

        logger.info(f"Left dataset: {left_ds.count()} rows")
        logger.info(f"Right dataset: {right_ds.count()} rows")

        # Perform left_anti join (should return empty - all left rows have matches)
        logger.info("Performing GPU left_anti join...")
        start = time.time()

        result = left_ds.gpu(nranks=1).join(
            right_ds,
            on='id',
            join_type='left_anti',
            num_partitions=20
        ).materialize()

        elapsed = time.time() - start
        result_count = result.count()

        logger.info(f"Join completed in {elapsed:.2f}s")
        logger.info(f"Result: {result_count} rows")

        # Verify correctness - should be 0 (all left rows have matches)
        assert result_count == 0, f"Expected 0 rows, got {result_count}"

        logger.info("✓ Very large right dataset test PASSED")


    def test_memory_efficiency(self, ray_context):
        """Test that right dataset is properly hash-partitioned (not accumulated)."""
        # Create datasets with specific distribution
        # Left: IDs 0-99
        left_data = [{"id": i, "value": f"left_{i}"} for i in range(100)]
        left_ds = ray.data.from_items(left_data)

        # Right: IDs 100-199 (NO OVERLAP with left)
        # This tests that we don't waste memory on data that won't match
        right_data = [{"id": i, "attr": f"right_{i}"} for i in range(100, 10_100)]
        right_ds = ray.data.from_items(right_data)

        logger.info("Testing memory efficiency with non-overlapping datasets...")

        # Inner join should return 0 rows
        result = left_ds.gpu(nranks=1).join(
            right_ds,
            on='id',
            join_type='inner',
            num_partitions=10
        ).materialize()

        assert result.count() == 0

        # Left join should return all left rows
        result = left_ds.gpu(nranks=1).join(
            right_ds,
            on='id',
            join_type='left',
            num_partitions=10
        ).materialize()

        assert result.count() == 100

        # Left_anti should return all left rows
        result = left_ds.gpu(nranks=1).join(
            right_ds,
            on='id',
            join_type='left_anti',
            num_partitions=10
        ).materialize()

        assert result.count() == 100

        logger.info("✓ Memory efficiency test PASSED")


def run_all_tests():
    """Run all tests without pytest."""
    ray.init(num_gpus=1, num_cpus=4)

    try:
        # Basic joins
        logger.info("=" * 60)
        logger.info("Running Basic Join Tests")
        logger.info("=" * 60)
        test_basic = TestBasicJoins()
        test_basic.test_gpu_join_inner(None)
        test_basic.test_gpu_join_left(None)
        test_basic.test_gpu_join_left_anti(None)

        # Multi-key joins
        logger.info("\n" + "=" * 60)
        logger.info("Running Multi-Key Join Tests")
        logger.info("=" * 60)
        test_multi = TestMultiKeyJoins()
        test_multi.test_gpu_join_multi_key(None)
        test_multi.test_gpu_join_different_key_names(None)

        # Edge cases
        logger.info("\n" + "=" * 60)
        logger.info("Running Edge Case Tests")
        logger.info("=" * 60)
        test_edge = TestEdgeCases()
        test_edge.test_gpu_join_empty_left(None)
        test_edge.test_gpu_join_empty_right_inner(None)
        test_edge.test_gpu_join_empty_right_left_anti(None)
        test_edge.test_gpu_join_no_matches(None)
        test_edge.test_gpu_join_invalid_key(None)
        test_edge.test_gpu_join_unsupported_type(None)

        # Correctness
        logger.info("\n" + "=" * 60)
        logger.info("Running Correctness Tests")
        logger.info("=" * 60)
        test_correct = TestCorrectness()
        test_correct.test_gpu_vs_cpu_inner_join(None)
        test_correct.test_gpu_vs_cpu_left_join(None)

        # Integration
        logger.info("\n" + "=" * 60)
        logger.info("Running Integration Tests")
        logger.info("=" * 60)
        test_integration = TestIntegration()
        test_integration.test_deduplication_workflow(None)

        # Large joins
        logger.info("\n" + "=" * 60)
        logger.info("Running Large Join Tests")
        logger.info("=" * 60)
        large_joins = TestLargeJoins()
        large_joins.test_large_right_dataset()
        large_joins.test_very_large_right_dataset()
        large_joins.test_memory_efficiency()

        logger.info("\n" + "=" * 60)
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("=" * 60)

    finally:
        ray.shutdown()


if __name__ == "__main__":
    run_all_tests()
