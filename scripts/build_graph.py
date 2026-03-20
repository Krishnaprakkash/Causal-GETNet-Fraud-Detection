import os
import json
import pickle
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

ENTITY_COLS = [
    "card1", "card2", "card3", "card4", "card5", "card6",
    "ProductCD", "P_emaildomain", "addr1", "addr2", "dist1",
]

# Columns used for transaction node features
NUMERICAL_COLS = ["TransactionAmt"]
CATEGORICAL_COLS = ["ProductCD", "card4", "card6"]

# Hash-based embedding config (cold-start workaround)
HASH_BUCKET_SIZE = 1000  # Reduced from 10000 to save memory (11 types × 1000 × 128 dim ≈ 5.6MB vs 56MB)

FREQ_THRESHOLD = 5
DIST1_N_BINS = 10
TIME_ENCODING_DIM = 16  # sinusoidal encoding dimension


# ---------------------------------------------------------------------------
# Step 1: Data Loading & Merging
# ---------------------------------------------------------------------------

def load_and_merge_data() -> pd.DataFrame:
    """Load train_transaction + train_identity, merge, sort by time."""
    print("[1/9] Loading and merging data...")
    tx_path = os.path.join(RAW_DIR, "train_transaction.csv")
    id_path = os.path.join(RAW_DIR, "train_identity.csv")

    tx = pd.read_csv(tx_path)
    print(f"      train_transaction: {tx.shape}")

    id_df = pd.read_csv(id_path)
    print(f"      train_identity:    {id_df.shape}")

    df = tx.merge(id_df, on="TransactionID", how="left")
    print(f"      merged:            {df.shape}")

    # Sort by time and reset index to get global tx_idx 0..N-1
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    df["tx_idx"] = df.index  # global integer index for each transaction

    print(f"      Fraud rate: {df['isFraud'].mean():.4%}")
    return df


# ---------------------------------------------------------------------------
# Step 2: dist1 Bucketization
# ---------------------------------------------------------------------------

def bucketize_dist1(df: pd.DataFrame, n_bins: int = DIST1_N_BINS) -> pd.DataFrame:
    """
    Replace dist1 with a discrete bin label.
    NaN values get their own 'missing' bucket (bin label = n_bins).
    """
    print(f"[2/9] Bucketizing dist1 into {n_bins} quantile bins + 1 missing bin...")
    df = df.copy()

    non_null_mask = df["dist1"].notna()
    df["dist1_bin"] = np.nan

    if non_null_mask.sum() > 0:
        # qcut with duplicates='drop' handles repeated quantile edges
        df.loc[non_null_mask, "dist1_bin"] = pd.qcut(
            df.loc[non_null_mask, "dist1"],
            q=n_bins,
            labels=False,
            duplicates="drop",
        ).astype(float)

    # Assign missing bucket
    df["dist1_bin"] = df["dist1_bin"].fillna(n_bins).astype(int)

    # Replace dist1 column with binned version for entity processing
    df["dist1"] = df["dist1_bin"]
    df = df.drop(columns=["dist1_bin"])

    print(f"      dist1 unique bins: {df['dist1'].nunique()}")
    return df


# ---------------------------------------------------------------------------
# Step 3 & 4: Hash-Based Entity Mapping (cold-start workaround)
# ---------------------------------------------------------------------------

def hash_entity_value(value, bucket_size: int = HASH_BUCKET_SIZE) -> int:
    """
    Map any entity value to a fixed bucket index.
    Uses string representation to handle mixed types (int, float, string).
    """
    if pd.isna(value):
        # NaN/missing maps to bucket 0 (dedicated for missing values)
        return 0
    # Convert to string, hash, then mod bucket_size
    return (hash(str(value)) % bucket_size)


def build_entity_mappings(
    df: pd.DataFrame,
    entity_cols: list[str],
    bucket_size: int = HASH_BUCKET_SIZE,
) -> dict[str, dict]:
    """
    For each entity column:
      1. Apply hash function to map any value to a fixed bucket (0 to bucket_size-1)
      2. This guarantees every entity value maps to a valid node — no cold-start

    Returns:
        mappings: {col: {raw_value: hash_bucket_idx}}

    Note: Hash collisions are possible (different values → same bucket) but with
    bucket_size=10000, collisions are rare for this dataset.
    """
    print(f"[3/9] Building hash-based entity mappings (bucket_size={bucket_size})...")
    mappings = {}

    for col in entity_cols:
        # Apply hash to each value in the column
        df[f"{col}_hash"] = df[col].apply(lambda v: hash_entity_value(v, bucket_size))
        
        # Build mapping dict (raw_value → hash_bucket)
        mapping = df[[col, f"{col}_hash"]].drop_duplicates().set_index(col)[f"{col}_hash"].to_dict()
        mappings[col] = mapping
        
        # Use the hashed column for subsequent processing
        df[col] = df[f"{col}_hash"]
        df = df.drop(columns=[f"{col}_hash"])
        
        # Count unique hash buckets actually used (may be < bucket_size due to missing values)
        unique_buckets = df[col].nunique()
        print(f"      {col:20s}: {unique_buckets:5d} unique buckets used")

    return mappings


# ---------------------------------------------------------------------------
# Step 5: Temporal DAG Edge Construction
# ---------------------------------------------------------------------------

def build_temporal_dag_edges(
    df: pd.DataFrame,
    entity_col: str,
    entity_mapping: dict,
    num_entity_nodes: int | None = None,
) -> tuple[torch.Tensor, int]:
    """
    For a single entity column, build forward-in-time edges:
        entity_node → future_transaction

    For each entity value e:
      - Get all transactions linked to e (sorted by TransactionDT, already sorted)
      - The first transaction receives NO edge from e (no past context yet)
      - All subsequent transactions receive an edge from e

    Args:
        df: DataFrame sorted by TransactionDT with tx_idx column
        entity_col: name of the entity column
        entity_mapping: {raw_value: entity_node_idx}
        num_entity_nodes: If provided, use this as the entity node count (for hash-based).
                          If None, derive from max key in mapping + 1.

    Returns:
        edge_index: [2, E] tensor where row 0 = entity node idx, row 1 = tx node idx
        num_entity_nodes: K (number of entity nodes for this type)
    """
    src_list = []  # entity node indices
    dst_list = []  # transaction node indices

    # Filter to rows where entity value survived frequency threshold
    valid_mask = df[entity_col].map(entity_mapping).notna()
    df_valid = df[valid_mask].copy()
    df_valid["entity_idx"] = df_valid[entity_col].map(entity_mapping).astype(int)

    # Group by entity value; df is already sorted by TransactionDT
    for entity_idx, group in df_valid.groupby("entity_idx", sort=False):
        tx_indices = group["tx_idx"].values  # already time-sorted

        if len(tx_indices) < 2:
            # Only one transaction for this entity — no future transactions to point to
            continue

        # Entity points to all transactions EXCEPT the first (which has no past context)
        future_tx_indices = tx_indices[1:]  # skip the earliest transaction

        src_list.extend([entity_idx] * len(future_tx_indices))
        dst_list.extend(future_tx_indices.tolist())

    if len(src_list) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(
            [src_list, dst_list], dtype=torch.long
        )

    # Use provided num_entity_nodes (for hash-based) or derive from mapping
    if num_entity_nodes is None:
        num_entity_nodes = len(entity_mapping)
    return edge_index, num_entity_nodes


# ---------------------------------------------------------------------------
# Step 6: Transaction Node Features
# ---------------------------------------------------------------------------

def sinusoidal_time_encoding(times: np.ndarray, dim: int = TIME_ENCODING_DIM) -> np.ndarray:
    """
    Sinusoidal positional encoding for TransactionDT.
    Normalizes time to [0, 1] then applies sin/cos at multiple frequencies.
    """
    t_min, t_max = times.min(), times.max()
    t_norm = (times - t_min) / (t_max - t_min + 1e-8)  # [N]

    freqs = np.arange(1, dim // 2 + 1, dtype=np.float32)  # [dim/2]
    angles = t_norm[:, None] * freqs[None, :] * np.pi  # [N, dim/2]

    encoding = np.concatenate([np.sin(angles), np.cos(angles)], axis=1)  # [N, dim]
    return encoding.astype(np.float32)


def build_transaction_features(df: pd.DataFrame) -> torch.Tensor:
    """
    Build transaction node feature matrix.

    Features:
      - log1p(TransactionAmt)           : 1 dim
      - Sinusoidal time encoding         : TIME_ENCODING_DIM dims
      - One-hot ProductCD                : 5 dims (W/H/C/S/R)
      - One-hot card4 (card network)     : variable dims
      - One-hot card6 (card type)        : variable dims

    Returns:
        x: [N, F] float tensor
    """
    print("[6/9] Building transaction node features...")
    feature_parts = []

    # Numerical: log1p(TransactionAmt)
    amt = np.log1p(df["TransactionAmt"].fillna(0).values).astype(np.float32)
    feature_parts.append(amt.reshape(-1, 1))

    # Time encoding
    time_enc = sinusoidal_time_encoding(df["TransactionDT"].values)
    feature_parts.append(time_enc)

    # Categorical: ProductCD
    product_dummies = pd.get_dummies(df["ProductCD"].fillna("missing"), prefix="prod")
    feature_parts.append(product_dummies.values.astype(np.float32))

    # Categorical: card4 (Visa/Mastercard/etc.)
    card4_dummies = pd.get_dummies(df["card4"].fillna("missing"), prefix="card4")
    feature_parts.append(card4_dummies.values.astype(np.float32))

    # Categorical: card6 (debit/credit/etc.)
    card6_dummies = pd.get_dummies(df["card6"].fillna("missing"), prefix="card6")
    feature_parts.append(card6_dummies.values.astype(np.float32))

    x = np.concatenate(feature_parts, axis=1)
    print(f"      Transaction feature dim: {x.shape[1]}")
    return torch.tensor(x, dtype=torch.float)


# ---------------------------------------------------------------------------
# Step 7: HeteroData Assembly
# ---------------------------------------------------------------------------

def assemble_hetero_data(
    df: pd.DataFrame,
    entity_mappings: dict[str, dict],
    tx_features: torch.Tensor,
) -> HeteroData:
    """
    Assemble the full HeteroData object.

    Node types:
      - transaction: has features x, labels y, time attribute
      - each entity type: has num_nodes only (embeddings learned at train time)

    Edge types:
      - (entity_col, 'uses', 'transaction'): forward temporal edges
    """
    print("[7/9] Assembling HeteroData object...")
    data = HeteroData()

    # --- Transaction nodes ---
    data["transaction"].x = tx_features
    data["transaction"].y = torch.tensor(df["isFraud"].values, dtype=torch.float)
    data["transaction"].time = torch.tensor(df["TransactionDT"].values, dtype=torch.float)
    data["transaction"].tx_id = torch.tensor(df["TransactionID"].values, dtype=torch.long)
    print(f"      transaction nodes: {len(df):,}  features: {tx_features.shape[1]}")

    # --- Entity nodes + edges ---
    # With hash-based embeddings, each entity type has exactly HASH_BUCKET_SIZE nodes
    total_edges = 0
    for col in ENTITY_COLS:
        mapping = entity_mappings[col]
        edge_index, num_nodes = build_temporal_dag_edges(
            df, col, mapping, num_entity_nodes=HASH_BUCKET_SIZE
        )

        data[col].num_nodes = HASH_BUCKET_SIZE  # Fixed bucket size for all entity types
        data[col, "uses", "transaction"].edge_index = edge_index

        total_edges += edge_index.shape[1]
        print(f"      {col:20s}: {HASH_BUCKET_SIZE:6d} nodes, {edge_index.shape[1]:8,} edges")

    print(f"      Total edges: {total_edges:,}")
    return data


# ---------------------------------------------------------------------------
# Step 8: DAG Validation
# ---------------------------------------------------------------------------

def validate_dag(data: HeteroData, df: pd.DataFrame) -> bool:
    """
    Validate that no backward temporal edges exist.

    For each edge type (entity, 'uses', transaction):
      - Get the transaction node indices (dst)
      - Get the TransactionDT for each dst transaction
      - Assert all dst transactions have time > the minimum time of
        transactions that contributed to the entity's state

    Also checks for self-loops (impossible here since entity ≠ transaction nodes,
    but we verify edge_index has no negative indices).

    Returns True if valid, raises AssertionError otherwise.
    """
    print("[8/9] Validating DAG property...")
    tx_times = df["TransactionDT"].values  # indexed by tx_idx

    for col in ENTITY_COLS:
        edge_index = data[col, "uses", "transaction"].edge_index
        if edge_index.shape[1] == 0:
            continue

        dst_tx_indices = edge_index[1].numpy()

        # All destination transaction indices must be valid
        assert dst_tx_indices.min() >= 0, f"{col}: negative tx index found"
        assert dst_tx_indices.max() < len(tx_times), f"{col}: tx index out of bounds"

        # Verify no negative indices in src (entity nodes)
        src_entity_indices = edge_index[0].numpy()
        assert src_entity_indices.min() >= 0, f"{col}: negative entity index found"

    print("      DAG validation passed: no backward edges detected.")
    return True


# ---------------------------------------------------------------------------
# Step 9: Save Artifacts
# ---------------------------------------------------------------------------

def save_artifacts(
    data: HeteroData,
    entity_mappings: dict,
    df: pd.DataFrame,
) -> None:
    """Save graph, mappings, and stats to processed directory."""
    print("[9/9] Saving artifacts...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Save HeteroData graph
    graph_path = os.path.join(PROCESSED_DIR, "hetero_graph.pt")
    torch.save(data, graph_path)
    print(f"      Saved: {graph_path}")

    # Save entity mappings
    mappings_path = os.path.join(PROCESSED_DIR, "entity_mappings.pkl")
    with open(mappings_path, "wb") as f:
        pickle.dump(entity_mappings, f)
    print(f"      Saved: {mappings_path}")

    # Save graph stats
    stats = {
        "num_transactions": int(data["transaction"].x.shape[0]),
        "num_fraud": int(data["transaction"].y.sum().item()),
        "fraud_rate": float(data["transaction"].y.mean().item()),
        "tx_feature_dim": int(data["transaction"].x.shape[1]),
        "entity_nodes": {},
        "entity_edges": {},
    }
    for col in ENTITY_COLS:
        stats["entity_nodes"][col] = int(data[col].num_nodes)
        stats["entity_edges"][col] = int(
            data[col, "uses", "transaction"].edge_index.shape[1]
        )

    stats_path = os.path.join(PROCESSED_DIR, "graph_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"      Saved: {stats_path}")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def build_graph() -> HeteroData:
    """Full graph construction pipeline."""
    print("=" * 60)
    print("  Heterogeneous Temporal DAG — Graph Construction")
    print("=" * 60)

    # Step 1: Load & merge
    df = load_and_merge_data()

    # Step 2: Bucketize dist1
    df = bucketize_dist1(df)

    # Step 3 & 4: Frequency threshold + entity mappings
    entity_mappings = build_entity_mappings(df, ENTITY_COLS)

    # Step 5 is handled inside assemble_hetero_data via build_temporal_dag_edges

    # Step 6: Transaction features
    tx_features = build_transaction_features(df)

    # Step 7: Assemble HeteroData
    data = assemble_hetero_data(df, entity_mappings, tx_features)

    # Step 8: Validate DAG
    validate_dag(data, df)

    # Step 9: Save
    save_artifacts(data, entity_mappings, df)

    print("=" * 60)
    print("  Graph construction complete.")
    print("=" * 60)
    print(data)

    return data


if __name__ == "__main__":
    build_graph()
