#!/usr/bin/env python3
"""
HeteroConv Inference for PJT2 Fraud Detection

Runs inference on test data using the trained HeteroConv model.
Generates fraud predictions and root cause analysis.

Input:  data/processed/best_model_heteroconv.pt
        data/processed/test_hetero_graph.pt (created by this script)
        data/raw/test_transaction.csv
        data/raw/test_identity.csv
Output: data/processed/fraud_predictions.csv
        data/processed/fraud_report.html

Configuration: Uses same config as training in scripts/config_heteroconv.py
"""

import os
import sys
import argparse
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, TransformerConv
from torch_geometric.data import HeteroData
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_heteroconv import MODEL_CONFIG

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

TRAIN_GRAPH_PATH = os.path.join(PROCESSED_DIR, "hetero_graph_with_env.pt")

MODEL_PATH = os.path.join(PROCESSED_DIR, "best_model_heteroconv.pt")
ENTITY_MAPPINGS_PATH = os.path.join(PROCESSED_DIR, "entity_mappings.pkl")
TEST_GRAPH_PATH = os.path.join(PROCESSED_DIR, "test_hetero_graph.pt")
PREDICTIONS_PATH = os.path.join(PROCESSED_DIR, "fraud_predictions.csv")
REPORT_PATH = os.path.join(PROCESSED_DIR, "fraud_report.html")
SAMPLE_SUBMISSION_PATH = os.path.join(RAW_DIR, "sample_submission.csv")

HASH_BUCKET_SIZE = 1000
DIST1_N_BINS = 10
TIME_ENCODING_DIM = 16
ENTITY_COLS = [
    "card1", "card2", "card3", "card4", "card5", "card6",
    "ProductCD", "P_emaildomain", "addr1", "addr2", "dist1",
]

HIDDEN_DIM = MODEL_CONFIG["hidden_dim"]
NUM_LAYERS = MODEL_CONFIG["num_layers"]
NUM_HEADS = MODEL_CONFIG["num_heads"]
DROPOUT = MODEL_CONFIG["dropout"]
NUM_ENTITY_NODES = MODEL_CONFIG["num_entity_nodes"]
BATCH_SIZE = MODEL_CONFIG["batch_size"]
HETEROCONV_HEADS = MODEL_CONFIG["heteroconv_heads"]
HETEROCONV_DROPOUT = MODEL_CONFIG["heteroconv_dropout"]
USE_HETEROCONV = MODEL_CONFIG["use_heteroconv"]

FRAUD_THRESHOLD = 0.8
# ---------------------------------------------------------------------------
# Model Definition (duplicated from train_heteroconv.py for inference)
# ---------------------------------------------------------------------------


class HeteroConvFraudDetector(nn.Module):
    def __init__(
        self,
        tx_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.3,
        heteroconv_heads: int = 4,
        heteroconv_dropout: float = 0.2,
        use_heteroconv: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.entity_types = ENTITY_COLS
        self.num_entity_types = len(ENTITY_COLS)
        self.use_heteroconv = use_heteroconv

        self.entity_embeddings = nn.ModuleDict({
            entity_type: nn.Embedding(NUM_ENTITY_NODES, hidden_dim)
            for entity_type in self.entity_types
        })

        self.tx_proj = nn.Linear(tx_feature_dim, hidden_dim)

        if use_heteroconv:
            conv_dict = {}
            for entity_type in self.entity_types:
                conv_dict[(entity_type, "uses", "transaction")] = GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=heteroconv_heads,
                    concat=False,
                    dropout=heteroconv_dropout,
                    add_self_loops=False,
                )
            self.heteroconv = HeteroConv(conv_dict, aggr='sum')
            self.heteroconv_norm = nn.LayerNorm(hidden_dim)

        self.convs = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_channels = hidden_dim if layer_idx == 0 else hidden_dim * 2
            self.convs.append(
                TransformerConv(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    concat=(layer_idx < num_layers - 1),
                    dropout=dropout,
                )
            )

        self.batch_norms = nn.ModuleDict({
            entity_type: nn.LayerNorm(hidden_dim * num_heads if num_layers == 1 else hidden_dim)
            for entity_type in self.entity_types
        })

        classifier_input_dim = tx_feature_dim + hidden_dim * self.num_entity_types
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data, return_entity_contributions: bool = False):
        tx_x = data["transaction"].x
        tx_h = self.tx_proj(tx_x)

        x_dict = {"transaction": tx_h}
        for entity_type in self.entity_types:
            entity_x = data[entity_type].x
            entity_emb = self.entity_embeddings[entity_type](entity_x.squeeze(-1).long())
            x_dict[entity_type] = entity_emb

        edge_index_dict = {}
        for entity_type in self.entity_types:
            edge_type = (entity_type, "uses", "transaction")
            try:
                if edge_type in data.edge_index_dict:
                    edge_index_dict[edge_type] = data.edge_index_dict[edge_type]
            except (KeyError, AttributeError):
                pass

        entity_contributions = {}

        if self.use_heteroconv:
            out_dict = self.heteroconv(x_dict, edge_index_dict)
            for entity_type in self.entity_types:
                if entity_type in out_dict:
                    out_dict[entity_type] = self.heteroconv_norm(out_dict[entity_type])
        else:
            out_dict = {entity_type: x_dict[entity_type] for entity_type in self.entity_types}

        entity_messages_list = []

        for entity_type in self.entity_types:
            if entity_type in out_dict:
                h = out_dict[entity_type]
            else:
                entity_x = data[entity_type].x
                h = self.entity_embeddings[entity_type](entity_x.squeeze(-1).long())

            edge_type = (entity_type, "uses", "transaction")
            if edge_type in edge_index_dict:
                edge_index = edge_index_dict[edge_type]

                conv = self.convs[0]
                h = conv(h, edge_index)
                h = self.batch_norms[entity_type](h)

                num_tx = tx_x.size(0)
                agg_h = torch.zeros(num_tx, h.size(-1), device=h.device)

                src, dst = edge_index
                agg_h.index_add_(0, dst, h[src])

                if return_entity_contributions:
                    entity_contributions[entity_type] = agg_h.abs().mean(dim=-1)

                entity_messages_list.append(agg_h)

        if entity_messages_list:
            entity_messages = torch.cat(entity_messages_list, dim=-1)
        else:
            entity_messages = torch.zeros(tx_x.size(0), self.hidden_dim * self.num_entity_types, device=tx_x.device)

        combined = torch.cat([tx_x, entity_messages], dim=-1)
        logits = self.classifier(combined)

        if return_entity_contributions:
            return logits, entity_contributions
        return logits

    def get_entity_contributions(self, data):
        return self.forward(data, return_entity_contributions=True)


# ---------------------------------------------------------------------------
# Test Graph Building Functions
# ---------------------------------------------------------------------------


def hash_entity_value(value, bucket_size: int = HASH_BUCKET_SIZE) -> int:
    if pd.isna(value):
        return 0
    return (hash(str(value)) % bucket_size)


def sinusoidal_time_encoding(times: np.ndarray, dim: int = TIME_ENCODING_DIM) -> np.ndarray:
    t_min, t_max = times.min(), times.max()
    t_norm = (times - t_min) / (t_max - t_min + 1e-8)

    freqs = np.arange(1, dim // 2 + 1, dtype=np.float32)
    angles = t_norm[:, None] * freqs[None, :] * np.pi

    encoding = np.concatenate([np.sin(angles), np.cos(angles)], axis=1)
    return encoding.astype(np.float32)


def load_test_data():
    print("[1/6] Loading test data...")
    tx_path = os.path.join(RAW_DIR, "test_transaction.csv")
    id_path = os.path.join(RAW_DIR, "test_identity.csv")

    tx = pd.read_csv(tx_path)
    print(f"      test_transaction: {tx.shape}")

    id_df = pd.read_csv(id_path)
    print(f"      test_identity:    {id_df.shape}")

    df = tx.merge(id_df, on="TransactionID", how="left")
    print(f"      merged:            {df.shape}")

    df = df.sort_values("TransactionDT").reset_index(drop=True)
    df["tx_idx"] = df.index

    return df


def preprocess_test_data(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/6] Preprocessing test data...")
    df = df.copy()

    non_null_mask = df["dist1"].notna()
    df["dist1_bin"] = np.nan

    if non_null_mask.sum() > 0:
        df.loc[non_null_mask, "dist1_bin"] = pd.qcut(
            df.loc[non_null_mask, "dist1"],
            q=DIST1_N_BINS,
            labels=False,
            duplicates="drop",
        ).astype(float)

    df["dist1_bin"] = df["dist1_bin"].fillna(DIST1_N_BINS).astype(int)
    df["dist1"] = df["dist1_bin"]
    df = df.drop(columns=["dist1_bin"])

    for col in ENTITY_COLS:
        df[f"{col}_hash"] = df[col].apply(lambda v: hash_entity_value(v, HASH_BUCKET_SIZE))
        df[col] = df[f"{col}_hash"]
        df = df.drop(columns=[f"{col}_hash"])

    return df


def build_test_transaction_features(df: pd.DataFrame) -> torch.Tensor:
    print("[3/6] Building transaction features...")
    feature_parts = []

    amt = np.log1p(df["TransactionAmt"].fillna(0).values).astype(np.float32)
    feature_parts.append(amt.reshape(-1, 1))

    time_enc = sinusoidal_time_encoding(df["TransactionDT"].values)
    feature_parts.append(time_enc)

    product_dummies = pd.get_dummies(df["ProductCD"].fillna("missing"), prefix="prod")
    feature_parts.append(product_dummies.values.astype(np.float32))

    card4_dummies = pd.get_dummies(df["card4"].fillna("missing"), prefix="card4")
    feature_parts.append(card4_dummies.values.astype(np.float32))

    card6_dummies = pd.get_dummies(df["card6"].fillna("missing"), prefix="card6")
    feature_parts.append(card6_dummies.values.astype(np.float32))

    x = np.concatenate(feature_parts, axis=1)
    print(f"      Feature dim: {x.shape[1]}")
    return torch.tensor(x, dtype=torch.float)


def build_test_graph(df: pd.DataFrame, tx_features: torch.Tensor) -> HeteroData:
    print("[4/6] Building test graph...")
    data = HeteroData()

    data["transaction"].x = tx_features
    data["transaction"].tx_id = torch.tensor(df["TransactionID"].values, dtype=torch.long)
    data["transaction"].time = torch.tensor(df["TransactionDT"].values, dtype=torch.float)
    print(f"      Transaction nodes: {len(df):,}")

    total_edges = 0
    for col in ENTITY_COLS:
        src_list = []
        dst_list = []

        df_valid = df[df[col].notna()].copy()
        df_valid["entity_idx"] = df_valid[col].astype(int)

        for entity_idx, group in df_valid.groupby("entity_idx", sort=False):
            tx_indices = group["tx_idx"].values

            if len(tx_indices) < 2:
                continue

            future_tx_indices = tx_indices[1:]
            src_list.extend([entity_idx] * len(future_tx_indices))
            dst_list.extend(future_tx_indices.tolist())

        if len(src_list) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

        data[col].num_nodes = HASH_BUCKET_SIZE
        data[col, "uses", "transaction"].edge_index = edge_index

        total_edges += edge_index.shape[1]
        print(f"      {col:20s}: {HASH_BUCKET_SIZE:6d} nodes, {edge_index.shape[1]:8,} edges")

    print(f"      Total edges: {total_edges:,}")
    return data


# ---------------------------------------------------------------------------
# Inference Functions
# ---------------------------------------------------------------------------


def get_subgraph_for_batch(data, batch_tx_indices, device):
    subgraph = HeteroData()

    subgraph['transaction'].x = data['transaction'].x[batch_tx_indices].to(device)
    subgraph['transaction'].tx_id = data['transaction'].tx_id[batch_tx_indices].to(device)

    batch_tx_set = set(batch_tx_indices.cpu().tolist())
    batch_tx_indices_cpu = batch_tx_indices.cpu()

    for entity_type in ENTITY_COLS:
        edge_type = (entity_type, "uses", "transaction")
        try:
            has_edges = edge_type in data.edge_index_dict
        except (KeyError, AttributeError):
            has_edges = False

        if not has_edges:
            subgraph[entity_type].x = torch.tensor([[0]], dtype=torch.long, device=device)
            continue

        edge_index = data.edge_index_dict[edge_type]
        src = edge_index[0].cpu()
        dst = edge_index[1].cpu()

        mask = torch.tensor([dst_i in batch_tx_set for dst_i in dst], dtype=torch.bool)

        if mask.sum() > 0:
            connected_src = src[mask]
            connected_entity_indices = connected_src.unique().tolist()

            entity_idx_to_new = {idx: new_idx for new_idx, idx in enumerate(connected_entity_indices)}

            new_entity_indices = torch.tensor(connected_entity_indices, dtype=torch.long, device=device).unsqueeze(-1)
            subgraph[entity_type].x = new_entity_indices

            tx_idx_to_batch = {idx: i for i, idx in enumerate(batch_tx_indices_cpu)}

            filtered_edges = edge_index[:, mask]
            original_src = filtered_edges[0].cpu()
            original_dst = filtered_edges[1].cpu()

            new_src = torch.tensor([entity_idx_to_new[s.item()] for s in original_src], dtype=torch.long, device=device)
            new_dst = torch.tensor([tx_idx_to_batch[dst_i.item()] for dst_i in original_dst], dtype=torch.long, device=device)

            subgraph[edge_type].edge_index = torch.stack([new_src, new_dst])
        else:
            subgraph[entity_type].x = torch.tensor([[0]], dtype=torch.long, device=device)

    return subgraph


def run_inference(model, data, device, batch_size: int = 1024):
    print("[5/6] Running inference...")
    model.eval()

    num_tx = data["transaction"].x.size(0)
    all_preds = []
    all_entity_contribs = []
    all_tx_ids = []

    num_batches = (num_tx + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Inference"):
            start = batch_idx * batch_size
            end = min(start + batch_size, num_tx)
            batch_tx_indices = torch.arange(start, end, device=device)

            batch_data = get_subgraph_for_batch(data, batch_tx_indices, device)

            logits, entity_contribs = model.get_entity_contributions(batch_data)
            preds = torch.sigmoid(logits).squeeze(-1)

            all_preds.extend(preds.cpu().numpy())
            all_tx_ids.extend(batch_data["transaction"].tx_id.cpu().numpy())

            contrib_array = {}
            for entity_type, contrib in entity_contribs.items():
                contrib_array[entity_type] = contrib.cpu().numpy()
            all_entity_contribs.append(contrib_array)

            del batch_data

    tx_ids = all_tx_ids
    preds = np.array(all_preds)

    entity_contrib_sums = {col: np.zeros(num_tx) for col in ENTITY_COLS}
    for batch_contribs in all_entity_contribs:
        offset = 0
        for entity_type in ENTITY_COLS:
            if entity_type in batch_contribs:
                n = len(batch_contribs[entity_type])
                entity_contrib_sums[entity_type][offset:offset+n] += batch_contribs[entity_type]
                offset += n

    root_causes = []
    for i in range(num_tx):
        max_entity = None
        max_contrib = -1
        for entity_type in ENTITY_COLS:
            if entity_contrib_sums[entity_type][i] > max_contrib:
                max_contrib = entity_contrib_sums[entity_type][i]
                max_entity = entity_type
        root_causes.append(max_entity if max_entity else "unknown")

    return tx_ids, preds, root_causes


def compute_statistics(preds, root_causes, threshold: float = 0.5):
    flagged = preds >= threshold
    n_flagged = flagged.sum()
    n_total = len(preds)

    fraud_by_entity = {}
    for entity in ENTITY_COLS:
        fraud_by_entity[entity] = sum(1 for i, fc in enumerate(root_causes) if fc == entity and flagged[i])

    stats = {
        "total_transactions": n_total,
        "flagged_frauds": int(n_flagged),
        "fraud_percentage": float(n_flagged / n_total * 100),
        "threshold": threshold,
        "fraud_by_entity": fraud_by_entity,
    }

    return stats


def compute_dynamic_thresholds(preds, threshold=FRAUD_THRESHOLD):
    """Compute dynamic risk thresholds based on prediction distribution."""
    above_threshold = preds[preds >= threshold]
    
    if len(above_threshold) == 0:
        return threshold, threshold, threshold
    
    low_threshold = float(np.percentile(above_threshold, 50))
    high_threshold = float(np.percentile(above_threshold, 75))
    
    return threshold, low_threshold, high_threshold


def classify_risk_category(pred, fraud_threshold, medium_threshold, high_threshold):
    """Classify a prediction into risk category."""
    if pred >= high_threshold:
        return "High Risk"
    elif pred >= medium_threshold:
        return "Medium Risk"
    elif pred >= fraud_threshold:
        return "Low Risk"
    else:
        return "No Risk"


def generate_html_report(csv_path, stats):
    print("[6/6] Generating HTML report...")

    df = pd.read_csv(csv_path)

    risk_counts = df['risk_category'].value_counts()
    high_risk_count = risk_counts.get('High Risk', 0) or 0
    medium_risk_count = risk_counts.get('Medium Risk', 0) or 0
    low_risk_count = risk_counts.get('Low Risk', 0) or 0
    total_flagged = high_risk_count + medium_risk_count + low_risk_count

    preds = df['isFraud'].values
    fraud_threshold = FRAUD_THRESHOLD
    low_risk_threshold, medium_risk_threshold, high_risk_threshold = compute_dynamic_thresholds(preds, fraud_threshold)

    if total_flagged > 0:
        max_category = max(high_risk_count, medium_risk_count, low_risk_count)
        high_risk_percent = (high_risk_count / max_category * 100) if max_category > 0 else 0
        medium_risk_percent = (medium_risk_count / max_category * 100) if max_category > 0 else 0
        low_risk_percent = (low_risk_count / max_category * 100) if max_category > 0 else 0
    else:
        high_risk_percent = medium_risk_percent = low_risk_percent = 0

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fraud Detection Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #dc3545;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-box.danger {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}
        .stat-box.warning {{
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        }}
        .stat-value {{
            font-size: 32px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .stat-label {{
            font-size: 14px;
            opacity: 0.9;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
            color: #333;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .high-risk {{
            color: #dc3545;
            font-weight: bold;
        }}
        .medium-risk {{
            color: #ffc107;
            font-weight: bold;
        }}
        .low-risk {{
            color: #28a745;
        }}
        .chart-container {{
            margin: 30px 0;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
        }}
        .bar-chart {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .bar-row {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .bar-label {{
            width: 120px;
            font-size: 14px;
        }}
        .bar-track {{
            flex: 1;
            height: 30px;
            background-color: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
        }}
        .bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
        }}
        .bar-value {{
            width: 60px;
            text-align: right;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Fraud Detection Report</h1>
        <p>Generated by Causal-GETNet HeteroConv Model</p>
        
        <h2>Executive Summary</h2>
        <div class="summary-grid">
            <div class="stat-box">
                <div class="stat-label">Total Transactions</div>
                <div class="stat-value">{stats['total_transactions']:,}</div>
            </div>
            <div class="stat-box danger">
                <div class="stat-label">Flagged Frauds</div>
                <div class="stat-value">{stats['flagged_frauds']:,}</div>
            </div>
            <div class="stat-box warning">
                <div class="stat-label">Fraud Percentage</div>
                <div class="stat-value">{stats['fraud_percentage']:.2f}%</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h3>Flagged Transactions by Risk Category</h3>
            <div class="bar-chart">
                <div class="bar-row">
                    <div class="bar-label">High Risk</div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: {high_risk_percent}%; background: linear-gradient(90deg, #dc3545 0%, #f5576c 100%);"></div>
                    </div>
                    <div class="bar-value">{high_risk_count:,}</div>
                </div>
                <div class="bar-row">
                    <div class="bar-label">Medium Risk</div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: {medium_risk_percent}%; background: linear-gradient(90deg, #ffc107 0%, #fee140 100%);"></div>
                    </div>
                    <div class="bar-value">{medium_risk_count:,}</div>
                </div>
                <div class="bar-row">
                    <div class="bar-label">Low Risk</div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: {low_risk_percent}%; background: linear-gradient(90deg, #28a745 0%, #4caf50 100%);"></div>
                    </div>
                    <div class="bar-value">{low_risk_count:,}</div>
                </div>
            </div>
            <p style="margin-top: 15px;"><strong>Total Flagged:</strong> {total_flagged:,} transactions (fraud threshold: {fraud_threshold})</p>
            <p style="margin-top: 5px; font-size: 12px; color: #666;"><strong>Risk Thresholds (Dynamic):</strong> High: ≥{high_risk_threshold:.2f}, Medium: {fraud_threshold:.2f}-{high_risk_threshold:.2f}, Low: {fraud_threshold:.2f}-{medium_risk_threshold:.2f}</p>
        </div>
    </div>
</body>
</html>
"""

    return html_content


def main():
    parser = argparse.ArgumentParser(description="PJT2 HeteroConv Inference")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--threshold", type=float, default=FRAUD_THRESHOLD, help="Fraud threshold")
    args = parser.parse_args()

    print("=" * 60)
    print("  PJT2 HeteroConv Inference")
    print("=" * 60)

    if args.cpu:
        device = torch.device("cpu")
        print("Device: CPU (forced)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print(f"\n[1/2] Loading model from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    # Load training graph to get correct feature dimension
    print(f"Loading training graph from {TRAIN_GRAPH_PATH} to get feature dimension...")
    train_data = torch.load(TRAIN_GRAPH_PATH, map_location=device, weights_only=False)
    train_feature_dim = train_data["transaction"].x.size(1)
    print(f"Training graph feature dim: {train_feature_dim}")
    del train_data

    df = load_test_data()
    df = preprocess_test_data(df)

    try:
        with open(ENTITY_MAPPINGS_PATH, "rb") as f:
            entity_mappings = pickle.load(f)
    except FileNotFoundError:
        entity_mappings = {col: {} for col in ENTITY_COLS}

    if os.path.exists(TEST_GRAPH_PATH):
        print(f"Loading test graph from {TEST_GRAPH_PATH}...")
        data = torch.load(TEST_GRAPH_PATH, map_location=device, weights_only=False)
    else:
        tx_features = build_test_transaction_features(df)
        data = build_test_graph(df, tx_features)
        torch.save(data, TEST_GRAPH_PATH)
        print(f"Saved test graph to {TEST_GRAPH_PATH}")

    print(f"Test graph: {data['transaction'].x.size(0)} transactions")

    # Pad test features to match training feature dimension
    current_feature_dim = data["transaction"].x.size(1)
    if current_feature_dim < train_feature_dim:
        padding = torch.zeros(
            data["transaction"].x.size(0),
            train_feature_dim - current_feature_dim,
            device=device  # Use same device as data
        )
        data["transaction"].x = torch.cat([data["transaction"].x, padding], dim=1)
        print(f"Padded test features from {current_feature_dim} to {train_feature_dim}")

    feature_dim = train_feature_dim

    model = HeteroConvFraudDetector(
        tx_feature_dim=feature_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        heteroconv_heads=HETEROCONV_HEADS,
        heteroconv_dropout=HETEROCONV_DROPOUT,
        use_heteroconv=USE_HETEROCONV,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print(f"Model loaded (val_auc: {checkpoint.get('val_auc', 'N/A'):.4f})")

    tx_ids, preds, root_causes = run_inference(model, data, device, args.batch_size)

    stats = compute_statistics(preds, root_causes, args.threshold)

    fraud_threshold = FRAUD_THRESHOLD
    low_thresh, med_thresh, high_thresh = compute_dynamic_thresholds(preds, fraud_threshold)

    print(f"\nStatistics:")
    print(f"  Total transactions: {stats['total_transactions']:,}")
    print(f"  Flagged frauds: {stats['flagged_frauds']:,}")
    print(f"  Fraud percentage: {stats['fraud_percentage']:.2f}%")
    print(f"\nDynamic Risk Thresholds:")
    print(f"  High Risk:    ≥{high_thresh:.3f}")
    print(f"  Medium Risk:  {fraud_threshold:.2f} - {high_thresh:.3f}")
    print(f"  Low Risk:     {fraud_threshold:.2f} - {med_thresh:.3f}")

    predictions_df = pd.DataFrame({
        "TransactionID": tx_ids,
        "isFraud": preds,
        "risk_category": [classify_risk_category(p, fraud_threshold, med_thresh, high_thresh) for p in preds]
    })
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"\nPredictions saved to: {PREDICTIONS_PATH}")

    html_report = generate_html_report(PREDICTIONS_PATH, stats)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html_report)
    print(f"Report saved to: {REPORT_PATH}")

    print("=" * 60)
    print("  Inference Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()