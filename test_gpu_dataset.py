"""
Test suite for dataset_extensions module.

Tests GPU/CPU mode switching via .gpu() and .cpu() methods.
"""

import sys
import unittest
from typing import Any

import ray
import ray.data

# Import to install extensions
from gpu_dataset import GPUDataset, _get_available_gpus


class TestDatasetExtensions(unittest.TestCase):
    """Test dataset extension methods."""

    @classmethod
    def setUpClass(cls):
        """Initialize Ray once for all tests."""
        if not ray.is_initialized():
            # Initialize with 2 virtual GPUs for testing
            ray.init(num_gpus=2, ignore_reinit_error=True)

    @classmethod
    def tearDownClass(cls):
        """Shutdown Ray after all tests."""
        if ray.is_initialized():
            ray.shutdown()

    def test_gpu_detection(self):
        """Test GPU detection with Ray initialized."""
        num_gpus = _get_available_gpus()
        self.assertIsInstance(num_gpus, int)
        self.assertGreaterEqual(num_gpus, 0)
        print(f"Detected {num_gpus} GPUs")

    def test_gpu_explicit_nranks(self):
        """Test .gpu() with explicit nranks."""
        ds = ray.data.range(100)
        gpu_ds = ds.gpu(nranks=2)

        self.assertIsInstance(gpu_ds, GPUDataset)
        self.assertEqual(gpu_ds.nranks, 2)
        print("✓ .gpu(nranks=2) works")

    def test_gpu_auto_detection(self):
        """Test .gpu() with auto-detection."""
        ds = ray.data.range(100)

        try:
            gpu_ds = ds.gpu()
            self.assertIsInstance(gpu_ds, GPUDataset)
            self.assertGreater(gpu_ds.nranks, 0)
            print(f"✓ .gpu() auto-detected {gpu_ds.nranks} GPUs")
        except ValueError as e:
            # Expected if no GPUs available
            self.assertIn("Cannot auto-detect GPUs", str(e))
            print("✓ .gpu() correctly raises ValueError when no GPUs available")

    def test_gpu_invalid_nranks_zero(self):
        """Test .gpu() with nranks=0 raises ValueError."""
        ds = ray.data.range(100)

        with self.assertRaises(ValueError) as ctx:
            ds.gpu(nranks=0)

        self.assertIn("positive integer", str(ctx.exception))
        print("✓ .gpu(nranks=0) raises ValueError")

    def test_gpu_invalid_nranks_negative(self):
        """Test .gpu() with negative nranks raises ValueError."""
        ds = ray.data.range(100)

        with self.assertRaises(ValueError) as ctx:
            ds.gpu(nranks=-1)

        self.assertIn("positive integer", str(ctx.exception))
        print("✓ .gpu(nranks=-1) raises ValueError")

    def test_gpu_double_wrap_protection(self):
        """Test that double-wrapping is prevented."""
        ds = ray.data.range(100)
        gpu_ds = ds.gpu(nranks=2)

        # Try to wrap again
        gpu_ds2 = gpu_ds.gpu(nranks=2)

        # Should return the same object
        self.assertIs(gpu_ds2, gpu_ds)
        print("✓ Double-wrapping prevented")

    def test_cpu_extracts_dataset(self):
        """Test .cpu() extracts underlying dataset."""
        ds = ray.data.range(100)
        gpu_ds = ds.gpu(nranks=2)
        cpu_ds = gpu_ds.cpu()

        self.assertIsInstance(cpu_ds, ray.data.Dataset)
        self.assertNotIsInstance(cpu_ds, GPUDataset)
        # Should be the same underlying dataset
        self.assertIs(cpu_ds, gpu_ds.dataset)
        print("✓ .cpu() extracts dataset correctly")

    def test_cpu_materialize_true(self):
        """Test .cpu(materialize=True) works."""
        ds = ray.data.range(100)
        gpu_ds = ds.gpu(nranks=2)
        cpu_ds = gpu_ds.cpu(materialize=True)

        self.assertIsInstance(cpu_ds, ray.data.Dataset)
        # Verify it's materialized (this is harder to test directly,
        # but at minimum it should not raise an error)
        print("✓ .cpu(materialize=True) works")

    def test_cpu_on_plain_dataset_noop(self):
        """Test .cpu() on plain Dataset is a no-op."""
        ds = ray.data.range(100)
        cpu_ds = ds.cpu()

        # Should return self
        self.assertIs(cpu_ds, ds)
        print("✓ .cpu() on plain Dataset is no-op")

    def test_cpu_on_plain_dataset_with_materialize(self):
        """Test .cpu(materialize=True) on plain Dataset."""
        ds = ray.data.range(100)
        cpu_ds = ds.cpu(materialize=True)

        self.assertIsInstance(cpu_ds, ray.data.Dataset)
        print("✓ .cpu(materialize=True) on plain Dataset works")

    def test_roundtrip_conversion(self):
        """Test Dataset → GPU → CPU roundtrip."""
        ds = ray.data.range(100)
        gpu_ds = ds.gpu(nranks=2)
        cpu_ds = gpu_ds.cpu()

        # Should get back the original dataset
        self.assertIs(cpu_ds, ds)
        print("✓ Roundtrip conversion works")

    def test_methods_exist(self):
        """Test that methods are installed on both classes."""
        self.assertTrue(hasattr(ray.data.Dataset, 'gpu'))
        self.assertTrue(hasattr(ray.data.Dataset, 'cpu'))
        self.assertTrue(hasattr(GPUDataset, 'cpu'))
        print("✓ All methods installed")

    def test_integration_with_operations(self):
        """Integration test: use GPU dataset with actual operations."""
        # Create a simple dataset
        ds = ray.data.range(100).map(lambda x: {'id': x, 'value': x * 2})

        # Convert to GPU
        gpu_ds = ds.gpu(nranks=2)

        # Note: We can't actually test groupby().map_groups() here without
        # GPU-compatible functions, but we can verify the dataset is created
        self.assertIsInstance(gpu_ds, GPUDataset)
        self.assertEqual(gpu_ds.nranks, 2)

        # Convert back to CPU
        cpu_ds = gpu_ds.cpu()
        self.assertIsInstance(cpu_ds, ray.data.Dataset)

        print("✓ Integration test passed")


def run_tests():
    """Run all tests with verbose output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDatasetExtensions)

    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
