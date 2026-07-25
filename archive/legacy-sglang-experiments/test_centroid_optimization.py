"""
Test script demonstrating the mathematical equivalence:
    mean(x @ W_k) = mean(x) @ W_k

And showing how to use the optimized centroid computation.
"""

import torch
import sys
sys.path.insert(0, "/vast/projects/liuv/pennnetworks/jiaheng/sglang_my/python")

from sglang.srt.layers.attention.tree_sparse.tree_parser import FlatChunk
from sglang.srt.layers.attention.tree_sparse.centroid_manager import CentroidManager


def test_mathematical_equivalence():
    """
    Verify that mean(x @ W) = mean(x) @ W
    """
    print("="*80)
    print("MATHEMATICAL EQUIVALENCE TEST")
    print("="*80)

    # Setup
    seq_len = 100
    hidden_dim = 512
    num_kv_heads = 8
    head_dim = 64
    kv_dim = num_kv_heads * head_dim

    # Create random hidden states and weight matrix
    torch.manual_seed(42)
    x = torch.randn(seq_len, hidden_dim, dtype=torch.float32)
    W_k = torch.randn(hidden_dim, kv_dim, dtype=torch.float32)

    # Define a chunk (e.g., tokens 20-50)
    chunk_start = 20
    chunk_end = 50
    chunk_x = x[chunk_start:chunk_end+1]  # [31, hidden_dim]

    print(f"\nChunk size: {chunk_x.shape[0]} tokens")
    print(f"Hidden dim: {hidden_dim}")
    print(f"KV dim: {kv_dim} (num_kv_heads={num_kv_heads}, head_dim={head_dim})")

    # Method 1: Project then average (current approach)
    print("\n" + "-"*80)
    print("Method 1: Project all tokens, then compute mean")
    print("-"*80)
    K_chunk = chunk_x @ W_k  # [31, kv_dim]
    centroid_method1 = K_chunk.mean(dim=0)  # [kv_dim]
    print(f"Projected {chunk_x.shape[0]} tokens: {K_chunk.shape}")
    print(f"Centroid shape: {centroid_method1.shape}")
    print(f"Centroid mean: {centroid_method1.mean().item():.6f}")
    print(f"Centroid std: {centroid_method1.std().item():.6f}")

    # Method 2: Average then project (optimized approach)
    print("\n" + "-"*80)
    print("Method 2: Compute mean of hidden states, then project")
    print("-"*80)
    x_mean = chunk_x.mean(dim=0)  # [hidden_dim]
    centroid_method2 = x_mean @ W_k  # [kv_dim]
    print(f"Averaged hidden states: {x_mean.shape}")
    print(f"Projected 1 centroid: {centroid_method2.shape}")
    print(f"Centroid mean: {centroid_method2.mean().item():.6f}")
    print(f"Centroid std: {centroid_method2.std().item():.6f}")

    # Compare
    print("\n" + "-"*80)
    print("COMPARISON")
    print("-"*80)
    diff = (centroid_method1 - centroid_method2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Max absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")
    print(f"Are they equal (within 1e-5)? {max_diff < 1e-5}")

    # Computational savings
    print("\n" + "-"*80)
    print("COMPUTATIONAL SAVINGS")
    print("-"*80)
    projections_method1 = chunk_x.shape[0]  # Project every token
    projections_method2 = 1  # Project only the centroid
    speedup = projections_method1 / projections_method2

    print(f"Method 1: {projections_method1} projections (one per token)")
    print(f"Method 2: {projections_method2} projection (just the centroid)")
    print(f"Speedup: {speedup}x fewer projections")

    assert max_diff < 1e-5, f"Methods don't match! Max diff: {max_diff}"
    print("\n✓ TEST PASSED: Methods are mathematically equivalent!")


def test_full_sequence_with_chunks():
    """
    Test with multiple chunks across a full sequence.
    """
    print("\n\n" + "="*80)
    print("FULL SEQUENCE TEST WITH MULTIPLE CHUNKS")
    print("="*80)

    # Setup
    seq_len = 10000
    hidden_dim = 5120
    num_kv_heads = 128
    head_dim = 128
    kv_dim = num_kv_heads * head_dim

    # Create random data
    torch.manual_seed(42)
    x = torch.randn(seq_len, hidden_dim, dtype=torch.float32)
    W_k = torch.randn(hidden_dim, kv_dim, dtype=torch.float32)

    # Create chunks (simulating tree-based chunking)
    chunks = [
        FlatChunk(0, 99, 0, "system"),
        FlatChunk(100, 599, 1, "user"),
        FlatChunk(600, 1599, 2, "html"),
        FlatChunk(1600, 2599, 3, "body"),
        FlatChunk(2600, 4999, 4, "content"),
        FlatChunk(5000, 9999, 5, "more_content"),
    ]

    print(f"\nSequence length: {seq_len}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"KV heads: {num_kv_heads}, head dim: {head_dim}")

    # Method 1: Current approach (project all, then average per chunk)
    print("\n" + "-"*80)
    print("Method 1: Project all tokens ({:,} projections)".format(seq_len))
    print("-"*80)
    K_all = x @ W_k  # [seq_len, kv_dim] - EXPENSIVE!
    centroids_method1 = []
    for chunk in chunks:
        chunk_K = K_all[chunk.start_idx:chunk.end_idx+1]
        centroid = chunk_K.mean(dim=0)
        centroids_method1.append(centroid)
    centroids_method1 = torch.stack(centroids_method1)  # [num_chunks, kv_dim]
    print(f"Centroids shape: {centroids_method1.shape}")

    # Method 2: Optimized (average per chunk, then project)
    print("\n" + "-"*80)
    print("Method 2: Project only centroids ({} projections)".format(len(chunks)))
    print("-"*80)
    centroids_method2 = []
    for chunk in chunks:
        chunk_x = x[chunk.start_idx:chunk.end_idx+1]
        x_mean = chunk_x.mean(dim=0)  # Average in hidden space
        centroid = x_mean @ W_k  # Project the average
        centroids_method2.append(centroid)
    centroids_method2 = torch.stack(centroids_method2)  # [num_chunks, kv_dim]
    print(f"Centroids shape: {centroids_method2.shape}")

    # Compare
    print("\n" + "-"*80)
    print("COMPARISON")
    print("-"*80)
    diff = (centroids_method1 - centroids_method2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Max absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")

    # Savings
    print("\n" + "-"*80)
    print("COMPUTATIONAL SAVINGS")
    print("-"*80)
    projections_method1 = seq_len
    projections_method2 = len(chunks)
    speedup = projections_method1 / projections_method2

    print(f"Method 1: {projections_method1:,} projections")
    print(f"Method 2: {projections_method2} projections")
    print(f"Speedup: {speedup:.0f}x fewer projections")
    print(f"Memory for K: {K_all.element_size() * K_all.nelement() / 1024**2:.1f} MB")

    # Tolerance adjusted for float32 accumulation errors with large matrices
    assert max_diff < 1e-4, f"Methods don't match! Max diff: {max_diff}"
    print(f"\n✓ TEST PASSED: Multi-chunk computation is equivalent (within {max_diff:.2e})!")


def test_centroid_manager_integration():
    """
    Test the CentroidManager with the new method.
    """
    print("\n\n" + "="*80)
    print("CENTROID MANAGER INTEGRATION TEST")
    print("="*80)

    # Setup
    num_layers = 32
    num_kv_heads = 128
    head_dim = 128
    hidden_dim = 5120
    seq_len = 5000
    kv_dim = num_kv_heads * head_dim

    # Create manager
    manager = CentroidManager(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device="cpu",
        dtype=torch.float16,
    )

    # Create chunks and register
    chunks = [
        FlatChunk(0, 99, 0, "system"),
        FlatChunk(100, 999, 1, "user"),
        FlatChunk(1000, 4999, 2, "content"),
    ]

    req_pool_idx = 0
    manager.register_request(req_pool_idx, chunks)

    # Create fake data
    torch.manual_seed(42)
    hidden_states = torch.randn(seq_len, hidden_dim, dtype=torch.float32)
    W_k = torch.randn(hidden_dim, kv_dim, dtype=torch.float32)

    # Use the optimized method
    layer_id = 0
    print(f"\nComputing centroids from hidden states...")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of chunks: {len(chunks)}")
    print(f"  Projections needed: {len(chunks)} (instead of {seq_len})")

    manager.update_centroids_from_hidden_states(
        req_pool_idx=req_pool_idx,
        layer_id=layer_id,
        hidden_states=hidden_states,
        W_k=W_k,
        seq_len=seq_len,
    )

    # Retrieve centroids
    result = manager.get_centroids(req_pool_idx, layer_id)
    assert result is not None, "Failed to retrieve centroids"

    centroids, retrieved_chunks = result
    print(f"\n✓ Centroids computed successfully!")
    print(f"  Shape: {centroids.shape}")
    print(f"  Dtype: {centroids.dtype}")
    print(f"  Chunks retrieved: {len(retrieved_chunks)}")

    # Verify against standard method
    K_all = hidden_states @ W_k
    K_all_reshaped = K_all.view(seq_len, num_kv_heads, head_dim)

    expected_centroids = []
    for chunk in chunks:
        chunk_K = K_all_reshaped[chunk.start_idx:chunk.end_idx+1]
        expected_centroids.append(chunk_K.mean(dim=0))
    expected_centroids = torch.stack(expected_centroids).to(torch.float16)

    diff = (centroids - expected_centroids).abs().max().item()
    print(f"\n  Max difference from standard method: {diff:.2e}")
    # float16 has ~3 decimal digits of precision, so 0.01 is reasonable
    assert diff < 0.01, f"Mismatch: {diff}  (float16 quantization error)"

    print("\n✓ TEST PASSED: CentroidManager integration works!")


if __name__ == "__main__":
    test_mathematical_equivalence()
    test_full_sequence_with_chunks()
    test_centroid_manager_integration()

    print("\n" + "="*80)
    print("ALL TESTS PASSED! ✓")
    print("="*80)
    print("\nSummary:")
    print("  • Mathematical equivalence verified: mean(x @ W) = mean(x) @ W")
    print("  • Computational savings: ~1000x fewer projections for typical sequences")
    print("  • This enables sparse attention during PREFILL, not just decode")
    print("  • To use: call update_centroids_from_hidden_states() before QKV projection")
