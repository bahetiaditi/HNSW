"""
Download and verify the SIFT1M dataset from HuggingFace.

SIFT1M contains:
  - 1,000,000 base vectors (128-dim, float32) for indexing
  - 10,000 query vectors (128-dim, float32)
  - 100 ground-truth nearest neighbors per query (unfiltered)

Total download: ~570 MB.

Usage:
    python data/download_sift1m.py [--data-dir DATA_DIR]
"""

import argparse
import os
import sys

import numpy as np
from huggingface_hub import hf_hub_download

# HuggingFace dataset repo
HF_REPO = "qbo-odp/sift1m"

# Files to download and their expected properties after loading
FILES = {
    "sift_base.fvecs": {
        "expected_shape": (1_000_000, 128),
        "dtype": "float32",
        "description": "1M base vectors (128-dim)",
    },
    "sift_query.fvecs": {
        "expected_shape": (10_000, 128),
        "dtype": "float32",
        "description": "10K query vectors (128-dim)",
    },
    "sift_groundtruth.ivecs": {
        "expected_shape": (10_000, 100),
        "dtype": "int32",
        "description": "Ground truth: 100 nearest neighbors per query",
    },
}


def read_vecs(filepath: str, dtype: str = "float32") -> np.ndarray:
    """Read .fvecs or .ivecs binary format.

    Format: each vector is stored as [dim: int32] [v1: dtype] [v2: dtype] ... [v_dim: dtype]
    So each 128-dim float32 vector occupies 4 + 128*4 = 516 bytes.

    Args:
        filepath: Path to .fvecs or .ivecs file.
        dtype: 'float32' for .fvecs, 'int32' for .ivecs.

    Returns:
        numpy array of shape (n_vectors, dim).
    """
    np_dtype = np.float32 if dtype == "float32" else np.int32

    with open(filepath, "rb") as f:
        # Read first dimension value to determine vector size
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)

        # Read entire file as int32 to handle the dim prefix uniformly
        raw = np.fromfile(f, dtype=np.int32)

    # Each vector has (1 + dim) int32 values: [dim, v1, v2, ..., v_dim]
    # But for fvecs, the vector values are float32, so we need to reinterpret
    # Reshape: each row is (dim+1) int32 values, drop the first column (dim prefix)
    vectors_per_row = dim + 1
    n_vectors = raw.shape[0] // vectors_per_row
    raw = raw.reshape(n_vectors, vectors_per_row)

    # Drop the dim column (first column) and reinterpret as the target dtype
    vectors = raw[:, 1:].copy()
    if dtype == "float32":
        vectors = vectors.view(np.float32)

    return np.ascontiguousarray(vectors)


def download_sift1m(data_dir: str) -> None:
    """Download SIFT1M files from HuggingFace and verify shapes."""
    os.makedirs(data_dir, exist_ok=True)

    print(f"Downloading SIFT1M dataset to {data_dir}/")
    print(f"Source: huggingface.co/datasets/{HF_REPO}\n")

    for filename, meta in FILES.items():
        target_path = os.path.join(data_dir, filename)

        # Skip if already downloaded and verified
        if os.path.exists(target_path):
            print(f"  {filename} already exists, verifying...")
            if _verify_file(target_path, meta):
                print(f"  ✓ {filename} verified ({meta['description']})\n")
                continue
            else:
                print(f"  ✗ Verification failed, re-downloading...")
                os.remove(target_path)

        # Download from HuggingFace
        print(f"  Downloading {filename} ({meta['description']})...")
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=filename,
            repo_type="dataset",
            local_dir=data_dir,
        )

        # hf_hub_download may place the file in a subdirectory; move if needed
        if downloaded_path != target_path and os.path.exists(downloaded_path):
            os.rename(downloaded_path, target_path)

        # Verify
        if _verify_file(target_path, meta):
            print(f"  ✓ {filename} verified ({meta['description']})\n")
        else:
            print(f"  ✗ {filename} verification FAILED\n")
            sys.exit(1)

    print("All files downloaded and verified successfully.")
    print(f"Dataset location: {os.path.abspath(data_dir)}/")


def _verify_file(filepath: str, meta: dict) -> bool:
    """Verify a downloaded file has the expected shape."""
    try:
        vectors = read_vecs(filepath, dtype=meta["dtype"])
        if vectors.shape != meta["expected_shape"]:
            print(
                f"    Shape mismatch: got {vectors.shape}, "
                f"expected {meta['expected_shape']}"
            )
            return False
        return True
    except Exception as e:
        print(f"    Read error: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SIFT1M dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__)),
        help="Directory to store dataset files (default: data/)",
    )
    args = parser.parse_args()
    download_sift1m(args.data_dir)