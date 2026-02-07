"""
Test to compare CPU and GPU MinHash implementations for duplicate detection.

This test verifies that both generate_minhash_signatures (CPU) and
generate_minhash_signatures_gpu (GPU) find the same duplicates when
applied to the same dataset.
"""

import numpy as np
import pandas as pd
import ray
import pytest
from typing import Set, Tuple

# Try to import ray_data_dedup3 functions - handle GPU dependency errors
try:
    from ray_data_dedup3 import (
        generate_minhash_signatures,
        generate_minhash_signatures_gpu,
        generate_lsh_bands,
        optimal_param,
        create_edges_from_collisions,
        compute_connected_components_distributed,
    )
    IMPORTS_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)
    print(f"Warning: Could not import from ray_data_dedup3: {e}")

try:
    import cudf
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class TestMinHash:
    def create_test_dataset(self) -> pd.DataFrame:
        """Create a synthetic dataset with known duplicates."""
        texts = [
            # Group 1: Exact duplicates
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog",

            # Group 2: Very similar (>90% similar)
            "Machine learning is a subset of artificial intelligence",
            "Machine learning is a subset of artificial intelligence systems",
            "Machine learning is a subset of artificial intelligence technology",

            # Group 3: Moderately similar (~70-80% similar)
            "Python is a popular programming language for data science",
            "Python is a widely used programming language for data analysis",

            # Group 4: Unique documents
            "The weather today is sunny and warm",
            "Quantum computing uses quantum bits or qubits",
            "The history of Rome spans over two thousand years",

            # Group 5: Another set of near-duplicates
            "Deep learning neural networks require large datasets",
            "Deep learning neural networks need large datasets",

            # Group 6: Unique
            "The Pacific Ocean is the largest ocean on Earth",
            "Vincent van Gogh painted The Starry Night",
        ]

        return pd.DataFrame({
            'id': list(range(len(texts))),
            'text': texts,
        })

    def run_dedup_pipeline(
        self,
        df: pd.DataFrame,
        use_gpu: bool,
        threshold: float = 0.7,
        num_perm: int = 128,
        ngram_size: int = 5,
        seed: int = 42,
    ) -> Tuple[Set[int], Set[frozenset]]:
        """
        Run the full deduplication pipeline and return duplicate IDs and clusters.

        Returns:
            - Set of document IDs that are duplicates (node != parent)
            - Set of duplicate clusters (frozensets of doc IDs)
        """
        # Create Ray dataset
        ds = ray.data.from_pandas(df)
        ds = ds.materialize()

        # Step 1: Generate MinHash signatures
        if use_gpu:
            ds_with_minhash = ds.map_batches(
                generate_minhash_signatures_gpu,
                fn_kwargs={
                    'text_column': 'text',
                    'num_perm': num_perm,
                    'ngram_size': ngram_size,
                    'seed': seed,
                },
                batch_format='numpy',
                batch_size=100,  # Required when using GPUs
                num_gpus=1,
            )
        else:
            ds_with_minhash = ds.map_batches(
                generate_minhash_signatures,
                fn_kwargs={
                    'text_column': 'text',
                    'num_perm': num_perm,
                    'ngram_size': ngram_size,
                    'seed': seed,
                },
                batch_format='numpy',
            )

        ds_with_minhash = ds_with_minhash.materialize()

        # Step 2: Generate LSH bands
        num_bands, rows_per_band = optimal_param(threshold, num_perm)
        print(f"LSH parameters: {num_bands} bands, {rows_per_band} rows per band")

        bands_ds = ds_with_minhash.map_batches(
            generate_lsh_bands,
            fn_kwargs={
                'num_bands': num_bands,
                'rows_per_band': rows_per_band,
            },
            batch_format='numpy',
        )
        bands_ds = bands_ds.materialize()

        # Step 3: Create edges from collisions
        edges_ds = (
            bands_ds
            .groupby(['band_id', 'band_hash'], num_partitions=10)
            .map_groups(create_edges_from_collisions, batch_format="numpy")
            .materialize()
        )

        edges_count = edges_ds.count()
        print(f"Number of edges: {edges_count}")

        if edges_count == 0:
            return set(), set()

        # Step 4: Deduplicate edges
        edges_ds = edges_ds.groupby(['src', 'dst']).count().drop_columns(['count()']).materialize()

        # Step 5: Compute connected components
        edges_ds = edges_ds.rename_columns({"src": "node", "dst": "parent"}).materialize()
        components_ds = compute_connected_components_distributed(
            edges_ds,
            max_iterations=20,
            parallelism=10,
            num_gpus=1 if use_gpu else 0,
        )

        # Get results
        components_df = components_ds.to_pandas()

        # Find duplicate IDs (where node != parent)
        duplicate_ids = set(components_df[components_df['node'] != components_df['parent']]['node'])

        # Build duplicate clusters
        clusters = {}
        for _, row in components_df.iterrows():
            parent = row['parent']
            node = row['node']
            if parent not in clusters:
                clusters[parent] = set()
            clusters[parent].add(node)

        # Convert to frozensets and filter out singleton clusters
        duplicate_clusters = {frozenset(cluster) for cluster in clusters.values() if len(cluster) > 1}

        return duplicate_ids, duplicate_clusters

    def test_minhash_cpu_gpu_comparison(self):
        """Test that CPU and GPU MinHash implementations find the same duplicates."""
        if not IMPORTS_AVAILABLE:
            pytest.skip(f"Could not import required modules: {IMPORT_ERROR}")
        if not GPU_AVAILABLE:
            pytest.skip("GPU not available, skipping GPU test")

        # Create test dataset
        df = self.create_test_dataset()
        print(f"\nTest dataset size: {len(df)} documents")

        # Run deduplication with CPU
        print("\n" + "="*50)
        print("Running CPU-based deduplication...")
        print("="*50)
        cpu_duplicate_ids, cpu_clusters = self.run_dedup_pipeline(df, use_gpu=False)

        print(f"\nCPU Results:")
        print(f"  Duplicate IDs: {sorted(cpu_duplicate_ids)}")
        print(f"  Number of clusters: {len(cpu_clusters)}")
        print(f"  Clusters: {[sorted(list(c)) for c in cpu_clusters]}")

        # Run deduplication with GPU
        print("\n" + "="*50)
        print("Running GPU-based deduplication...")
        print("="*50)
        gpu_duplicate_ids, gpu_clusters = self.run_dedup_pipeline(df, use_gpu=True)

        print(f"\nGPU Results:")
        print(f"  Duplicate IDs: {sorted(gpu_duplicate_ids)}")
        print(f"  Number of clusters: {len(gpu_clusters)}")
        print(f"  Clusters: {[sorted(list(c)) for c in gpu_clusters]}")

        # Compare results
        print("\n" + "="*50)
        print("Comparison Results:")
        print("="*50)

        # Check if duplicate IDs match
        ids_match = cpu_duplicate_ids == gpu_duplicate_ids
        print(f"Duplicate IDs match: {ids_match}")

        if not ids_match:
            print(f"  Only in CPU: {cpu_duplicate_ids - gpu_duplicate_ids}")
            print(f"  Only in GPU: {gpu_duplicate_ids - cpu_duplicate_ids}")

        # Check if clusters match
        clusters_match = cpu_clusters == gpu_clusters
        print(f"Duplicate clusters match: {clusters_match}")

        if not clusters_match:
            print(f"  CPU-only clusters: {cpu_clusters - gpu_clusters}")
            print(f"  GPU-only clusters: {gpu_clusters - cpu_clusters}")

        # Calculate similarity metrics
        if cpu_duplicate_ids and gpu_duplicate_ids:
            jaccard = len(cpu_duplicate_ids & gpu_duplicate_ids) / len(cpu_duplicate_ids | gpu_duplicate_ids)
            print(f"Jaccard similarity of duplicate IDs: {jaccard:.3f}")

        # Assertions
        assert ids_match or len(cpu_duplicate_ids.symmetric_difference(gpu_duplicate_ids)) <= 2, \
            "CPU and GPU should find the same duplicates (or differ by at most 2 docs due to boundary cases)"

        print("\n✓ Test passed: CPU and GPU implementations produce consistent results")

    def test_minhash_signatures_format(self):
        """Test that CPU and GPU produce signatures in the same format."""
        if not IMPORTS_AVAILABLE:
            pytest.skip(f"Could not import required modules: {IMPORT_ERROR}")
        if not GPU_AVAILABLE:
            pytest.skip("GPU not available, skipping GPU test")

        if not ray.is_initialized():
            ray.init(num_gpus=1)

        # Create simple test batch
        batch = {
            'id': np.array([0, 1, 2]),
            'text': np.array([
                "The quick brown fox",
                "The quick brown fox",  # Duplicate
                "Something completely different",
            ])
        }

        num_perm = 128
        ngram_size = 5
        seed = 42

        # Generate CPU signatures
        cpu_result = generate_minhash_signatures(
            batch.copy(),
            text_column='text',
            num_perm=num_perm,
            ngram_size=ngram_size,
            seed=seed,
        )

        # Generate GPU signatures
        gpu_result = generate_minhash_signatures_gpu(
            batch.copy(),
            text_column='text',
            num_perm=num_perm,
            ngram_size=ngram_size,
            seed=seed,
        )

        # Check shapes
        assert cpu_result['minhash'].shape == (3, num_perm), "CPU signature shape incorrect"
        assert gpu_result['minhash'].shape == (3, num_perm), "GPU signature shape incorrect"

        # Check dtypes
        assert cpu_result['minhash'].dtype == np.uint32, "CPU signature dtype incorrect"
        assert gpu_result['minhash'].dtype == np.uint32, "GPU signature dtype incorrect"

        print("\n✓ Signature format test passed")

        # Check if duplicates have similar signatures
        cpu_sig_0 = cpu_result['minhash'][0]
        cpu_sig_1 = cpu_result['minhash'][1]
        cpu_sig_2 = cpu_result['minhash'][2]

        cpu_sim_01 = np.mean(cpu_sig_0 == cpu_sig_1)  # Should be high (duplicates)
        cpu_sim_02 = np.mean(cpu_sig_0 == cpu_sig_2)  # Should be low (different)

        gpu_sig_0 = gpu_result['minhash'][0]
        gpu_sig_1 = gpu_result['minhash'][1]
        gpu_sig_2 = gpu_result['minhash'][2]

        gpu_sim_01 = np.mean(gpu_sig_0 == gpu_sig_1)
        gpu_sim_02 = np.mean(gpu_sig_0 == gpu_sig_2)

        print(f"\nCPU: Duplicate similarity = {cpu_sim_01:.3f}, Different similarity = {cpu_sim_02:.3f}")
        print(f"GPU: Duplicate similarity = {gpu_sim_01:.3f}, Different similarity = {gpu_sim_02:.3f}")

        # Both should detect that doc 0 and 1 are more similar than 0 and 2
        assert cpu_sim_01 > cpu_sim_02, "CPU should detect duplicates"
        assert gpu_sim_01 > gpu_sim_02, "GPU should detect duplicates"

        print("✓ Both CPU and GPU correctly identify duplicates via signature similarity")


if __name__ == '__main__':
    test = TestMinHash()
    test.test_minhash_cpu_gpu_comparison()
    test.test_minhash_signatures_format()