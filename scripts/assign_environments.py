#!/usr/bin/env python3
"""
Environment Assignment Script for PJT2 Fraud Detection

Assigns 3 environment IDs to each transaction node for IRM training:
  - env_time: 8 time windows (distribution shift over time)
  - env_region: 5 geographic bins (based on addr1)
  - env_fraud_rate: 3 fraud rate regimes (low/medium/high)

Input:  PJT2/data/processed/hetero_graph.pt
Output: PJT2/data/processed/hetero_graph_with_env.pt
"""

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

GRAPH_INPUT_PATH = os.path.join(PROCESSED_DIR, "hetero_graph.pt")
GRAPH_OUTPUT_PATH = os.path.join(PROCESSED_DIR, "hetero_graph_with_env.pt")

N_TIME_WINDOWS = 8
N_REGIONS = 5
N_SEGMENTS = 10  # For computing fraud rate regimes

# ---------------------------------------------------------------------------
# Environment Computation Functions
# ---------------------------------------------------------------------------


def compute_env_time(times: np.ndarray, n_windows: int = N_TIME_WINDOWS) -> np.ndarray:
    """
    Compute env_time: divide transactions into n_windows equal-sample time windows.
    Uses percentile-based binning for equal sample counts.
    """
    print(f"\n[Computing env_time] {n_windows} time windows...")

    # Use percentile-based boundaries for equal sample counts
    percentiles = np.linspace(0, 100, n_windows + 1)
    boundaries = np.percentile(times, percentiles)

    # Handle duplicate boundaries (edge case)
    unique_boundaries = np.unique(boundaries)
    if len(unique_boundaries) < len(boundaries):
        print(f"      Warning: {len(boundaries) - len(unique_boundaries)} duplicate boundaries, using unique")
        boundaries = unique_boundaries

    # Digitize: assigns bin index 0 to n_windows-1
    env_time = np.digitize(times, boundaries[1:-1], right=True)

    # Ensure labels are in range 0 to n_windows-1
    env_time = np.clip(env_time, 0, n_windows - 1)

    print(f"      Time range: {times.min():.0f} to {times.max():.0f}")
    print(f"      Boundaries: {[f'{b:.0f}' for b in boundaries]}")

    return env_time


def compute_env_region(
    addr1_values: np.ndarray, n_regions: int = N_REGIONS
) -> np.ndarray:
    """
    Compute env_region: bin addr1 into n_regions quantile-based geographic bins.
    Missing values are assigned to bin 0.
    """
    print(f"\n[Computing env_region] {n_regions} geographic bins from addr1...")

    # Handle missing values: fill with -1, will go to first bin
    addr1_filled = np.where(pd.isna(addr1_values), -1, addr1_values)

    # Find non-missing values for quantile computation
    non_missing_mask = addr1_filled != -1
    n_non_missing = non_missing_mask.sum()
    n_missing = (~non_missing_mask).sum()

    print(f"      Non-missing addr1: {n_non_missing:,}")
    print(f"      Missing addr1: {n_missing:,}")

    # Initialize with 0 (will be correct for missing values)
    env_region = np.zeros(len(addr1_filled), dtype=np.int64)

    if n_non_missing > 0:
        # Get quantile boundaries from non-missing values
        try:
            # Use qcut for quantile-based binning
            # Handle duplicates by using 'drop' which may reduce number of bins
            valid_values = addr1_filled[non_missing_mask]
            _, bin_edges = pd.qcut(
                valid_values, q=n_regions, labels=False, retbins=True, duplicates="drop"
            )

            # Assign bins to non-missing values
            env_region[non_missing_mask] = np.digitize(
                valid_values, bin_edges[1:-1], right=True
            )

            # Clip to ensure valid range
            env_region = np.clip(env_region, 0, n_regions - 1)

            print(f"      Bin edges: {[f'{b:.2f}' for b in bin_edges]}")

        except ValueError as e:
            # Fallback: use rank-based binning
            print(f"      Warning: qcut failed ({e}), using rank-based binning")
            ranks = np.argsort(np.argsort(addr1_filled[non_missing_mask]))
            env_region[non_missing_mask] = (ranks * n_regions // n_non_missing).astype(
                np.int64
            )
            env_region = np.clip(env_region, 0, n_regions - 1)

    unique_bins = np.unique(env_region)
    print(f"      Actual number of bins used: {len(unique_bins)}")

    return env_region


def compute_env_fraud_rate(
    times: np.ndarray, fraud_labels: np.ndarray, n_segments: int = N_SEGMENTS
) -> np.ndarray:
    """
    Compute env_fraud_rate: divide timeline into segments, compute fraud rate per segment,
    then label as low (<2%), medium (2-5%), or high (>=5%).
    """
    print(f"\n[Computing env_fraud_rate] {n_segments} time segments -> 3 regimes...")

    # Divide into time segments
    percentiles = np.linspace(0, 100, n_segments + 1)
    boundaries = np.percentile(times, percentiles)

    segment_ids = np.digitize(times, boundaries[1:-1], right=True)
    segment_ids = np.clip(segment_ids, 0, n_segments - 1)

    # Compute fraud rate per segment
    fraud_rates = []
    for seg in range(n_segments):
        mask = segment_ids == seg
        if mask.sum() > 0:
            rate = fraud_labels[mask].mean()
        else:
            rate = 0.0
        fraud_rates.append(rate)
        print(f"      Segment {seg}: {mask.sum():,} txs, fraud rate = {rate:.4%}")

    # Assign regime labels based on fraud rate
    # Adjusted thresholds: 0 = low (<3%), 1 = medium (3-6%), 2 = high (>=6%)
    env_fraud_rate = np.zeros(len(times), dtype=np.int64)

    for seg_id, rate in enumerate(fraud_rates):
        mask = segment_ids == seg_id
        if rate < 0.03:
            env_fraud_rate[mask] = 0  # low
        elif rate < 0.06:
            env_fraud_rate[mask] = 1  # medium
        else:
            env_fraud_rate[mask] = 2  # high

    print(f"      Regime thresholds: low=<3%, medium=3-6%, high>=6%")

    return env_fraud_rate


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("Environment Assignment for IRM Training")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Step 1: Load the graph
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading graph from disk...")
    print(f"      Input:  {GRAPH_INPUT_PATH}")

    data: HeteroData = torch.load(GRAPH_INPUT_PATH, weights_only=False)
    n_transactions = data["transaction"].num_nodes

    print(f"      Loaded graph with {n_transactions:,} transaction nodes")

    # Extract transaction attributes
    times = data["transaction"].time.numpy()
    fraud_labels = data["transaction"].y.numpy()

    print(f"      Time attribute: shape={times.shape}, dtype={times.dtype}")
    print(f"      Fraud labels: shape={fraud_labels.shape}, dtype={fraud_labels.dtype}")
    print(f"      Overall fraud rate: {fraud_labels.mean():.4%}")

    # -------------------------------------------------------------------------
    # Step 2: Compute env_time
    # -------------------------------------------------------------------------
    env_time = compute_env_time(times, N_TIME_WINDOWS)

    # -------------------------------------------------------------------------
    # Step 3: Compute env_region (requires original addr1 data)
    # -------------------------------------------------------------------------
    print("\n[Step 3] Loading original data for addr1 column...")

    # Need to reload original transaction data to get addr1
    tx_path = os.path.join(RAW_DIR, "train_transaction.csv")

    # Read only necessary columns to save memory
    addr1_df = pd.read_csv(tx_path, usecols=["TransactionID", "addr1"])
    print(f"      Loaded addr1 from: {tx_path}")
    print(f"      Shape: {addr1_df.shape}")

    # Need to align addr1 with transaction nodes in the graph
    # The graph was sorted by TransactionDT, so we need to match by TransactionID
    # Get TransactionID from graph (stored as tx_id)
    tx_ids_in_graph = data["transaction"].tx_id.numpy()

    # Create mapping from TransactionID to addr1
    addr1_map = addr1_df.set_index("TransactionID")["addr1"].to_dict()

    # Get addr1 values in the same order as graph nodes
    addr1_values = np.array(
        [addr1_map.get(tx_id, np.nan) for tx_id in tx_ids_in_graph],
        dtype=np.float64
    )

    print(f"      Mapped addr1 for {len(addr1_values):,} transactions")

    env_region = compute_env_region(addr1_values, N_REGIONS)

    # -------------------------------------------------------------------------
    # Step 4: Compute env_fraud_rate
    # -------------------------------------------------------------------------
    env_fraud_rate = compute_env_fraud_rate(times, fraud_labels, N_SEGMENTS)

    # -------------------------------------------------------------------------
    # Step 5: Assign environment labels to HeteroData
    # -------------------------------------------------------------------------
    print("\n[Step 5] Assigning environment labels to HeteroData...")

    data["transaction"].env_time = torch.tensor(env_time, dtype=torch.long)
    data["transaction"].env_region = torch.tensor(env_region, dtype=torch.long)
    data["transaction"].env_fraud_rate = torch.tensor(env_fraud_rate, dtype=torch.long)

    print("      Added: data['transaction'].env_time")
    print("      Added: data['transaction'].env_region")
    print("      Added: data['transaction'].env_fraud_rate")

    # -------------------------------------------------------------------------
    # Step 6: Print environment distributions
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Environment Distributions")
    print("=" * 70)

    # env_time distribution
    print("\n[env_time] Distribution per time window:")
    for w in range(N_TIME_WINDOWS):
        mask = env_time == w
        count = mask.sum()
        fraud_rate = fraud_labels[mask].mean() if count > 0 else 0
        print(f"      Window {w}: {count:>7,} transactions ({count/len(env_time)*100:5.2f}%) | fraud rate: {fraud_rate:.4%}")

    # env_region distribution
    print("\n[env_region] Distribution per geographic bin:")
    for r in range(N_REGIONS):
        mask = env_region == r
        count = mask.sum()
        fraud_rate = fraud_labels[mask].mean() if count > 0 else 0
        print(f"      Bin {r}: {count:>7,} transactions ({count/len(env_region)*100:5.2f}%) | fraud rate: {fraud_rate:.4%}")

    # env_fraud_rate distribution
    regime_names = {0: "low (<3%)", 1: "medium (3-6%)", 2: "high (>=6%)"}
    print("\n[env_fraud_rate] Distribution per fraud rate regime:")
    for r in range(3):
        mask = env_fraud_rate == r
        count = mask.sum()
        fraud_rate = fraud_labels[mask].mean() if count > 0 else 0
        print(f"      {regime_names[r]:15s}: {count:>7,} transactions ({count/len(env_fraud_rate)*100:5.2f}%) | fraud rate: {fraud_rate:.4%}")

    # Verify fraud rate varies across environments
    print("\n[Validation] Fraud rate variation across environments:")
    
    # env_time variation
    env_time_rates = [fraud_labels[env_time == i].mean() for i in range(N_TIME_WINDOWS)]
    print(f"      env_time - range: {max(env_time_rates):.4%} to {min(env_time_rates):.4%}")
    
    # env_region variation
    env_region_rates = [fraud_labels[env_region == i].mean() for i in range(N_REGIONS)]
    print(f"      env_region - range: {max(env_region_rates):.4%} to {min(env_region_rates):.4%}")
    
    # env_fraud_rate variation (handle empty cases)
    low_mask = env_fraud_rate == 0
    med_mask = env_fraud_rate == 1
    high_mask = env_fraud_rate == 2
    
    low_rate = fraud_labels[low_mask].mean() if low_mask.sum() > 0 else float('nan')
    med_rate = fraud_labels[med_mask].mean() if med_mask.sum() > 0 else float('nan')
    high_rate = fraud_labels[high_mask].mean() if high_mask.sum() > 0 else float('nan')
    print(f"      env_fraud_rate - low: {low_rate:.4%}, med: {med_rate:.4%}, high: {high_rate:.4%}")

    # -------------------------------------------------------------------------
    # Step 7: Save updated graph
    # -------------------------------------------------------------------------
    print("\n[Step 7] Saving updated graph to disk...")
    print(f"      Output: {GRAPH_OUTPUT_PATH}")

    torch.save(data, GRAPH_OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("Environment assignment complete!")
    print("=" * 70)
    print(f"\nOutput saved to: {GRAPH_OUTPUT_PATH}")
    print("\nEnvironment labels are ready for IRM training:")
    print(f"  - data['transaction'].env_time       : shape {data['transaction'].env_time.shape}")
    print(f"  - data['transaction'].env_region     : shape {data['transaction'].env_region.shape}")
    print(f"  - data['transaction'].env_fraud_rate: shape {data['transaction'].env_fraud_rate.shape}")


if __name__ == "__main__":
    main()
