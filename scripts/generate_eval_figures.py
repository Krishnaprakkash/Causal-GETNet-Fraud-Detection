#!/usr/bin/env python3
"""
ML Evaluation Figures Generator

Generates comprehensive evaluation figures including:
- ROC AUC curves (comparing models)
- Precision-Recall curves
- Confusion matrices
- Threshold analysis plots

Input:  data/processed/best_model_heteroconv.pt
        data/processed/best_model.pt
        data/processed/hetero_graph_with_env.pt (contains labels)
Output: data/processed/figures/*.png
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, f1_score, precision_score, recall_score,
    roc_auc_score
)
from torch_geometric.nn import HeteroConv, GATConv, TransformerConv
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_heteroconv import MODEL_CONFIG as HETEROCONV_CONFIG

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(PROCESSED_DIR, "figures")

GRAPH_PATH = os.path.join(PROCESSED_DIR, "hetero_graph_with_env.pt")
HETEROCONV_MODEL_PATH = os.path.join(PROCESSED_DIR, "best_model_heteroconv.pt")
BASELINE_MODEL_PATH = os.path.join(PROCESSED_DIR, "best_model.pt")

HIDDEN_DIM = HETEROCONV_CONFIG["hidden_dim"]
NUM_LAYERS = HETEROCONV_CONFIG["num_layers"]
NUM_HEADS = HETEROCONV_CONFIG["num_heads"]
DROPOUT = HETEROCONV_CONFIG["dropout"]
NUM_ENTITY_NODES = HETEROCONV_CONFIG["num_entity_nodes"]
BATCH_SIZE = HETEROCONV_CONFIG["batch_size"]
HETEROCONV_HEADS = HETEROCONV_CONFIG["heteroconv_heads"]
HETEROCONV_DROPOUT = HETEROCONV_CONFIG["heteroconv_dropout"]
USE_HETEROCONV = HETEROCONV_CONFIG["use_heteroconv"]

ENTITY_TYPES = [
    "card1", "card2", "card3", "card4", "card5", "card6",
    "ProductCD", "P_emaildomain", "addr1", "addr2", "dist1",
]

TRAIN_RATIO = 0.8
DEFAULT_THRESHOLD = 0.5


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
        self.entity_types = ENTITY_TYPES
        self.num_entity_types = len(ENTITY_TYPES)
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

    def forward(self, data):
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
                entity_messages_list.append(agg_h)

        if entity_messages_list:
            entity_messages = torch.cat(entity_messages_list, dim=-1)
        else:
            entity_messages = torch.zeros(tx_x.size(0), self.hidden_dim * self.num_entity_types, device=tx_x.device)

        combined = torch.cat([tx_x, entity_messages], dim=-1)
        logits = self.classifier(combined)
        return logits


class BaselineFraudDetector(nn.Module):
    def __init__(
        self,
        tx_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.entity_types = ENTITY_TYPES
        self.num_entity_types = len(ENTITY_TYPES)

        self.entity_embeddings = nn.ModuleDict({
            entity_type: nn.Embedding(NUM_ENTITY_NODES, hidden_dim)
            for entity_type in self.entity_types
        })

        self.tx_proj = nn.Linear(tx_feature_dim, hidden_dim)

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

    def forward(self, data):
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

        entity_messages_list = []

        for entity_type in self.entity_types:
            h = x_dict[entity_type]

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
                entity_messages_list.append(agg_h)

        if entity_messages_list:
            entity_messages = torch.cat(entity_messages_list, dim=-1)
        else:
            entity_messages = torch.zeros(tx_x.size(0), self.hidden_dim * self.num_entity_types, device=tx_x.device)

        combined = torch.cat([tx_x, entity_messages], dim=-1)
        logits = self.classifier(combined)
        return logits


def get_subgraph_for_batch(data, batch_tx_indices, device):
    from torch_geometric.data import HeteroData
    
    subgraph = HeteroData()
    subgraph['transaction'].x = data['transaction'].x[batch_tx_indices].to(device)
    subgraph['transaction'].y = data['transaction'].y[batch_tx_indices].to(device)
    
    batch_tx_set = set(batch_tx_indices.cpu().tolist())
    batch_tx_indices_cpu = batch_tx_indices.cpu()
    
    for entity_type in ENTITY_TYPES:
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


def run_inference(model, data, val_idx, device, batch_size=1024):
    model.eval()
    all_preds = []
    all_labels = []
    
    num_val = len(val_idx)
    num_batches = (num_val + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, num_val)
            batch_tx_indices = val_idx[start:end]
            
            batch_data = get_subgraph_for_batch(data, batch_tx_indices, device)
            
            logits = model(batch_data)
            preds = torch.sigmoid(logits.squeeze(-1))
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_data["transaction"].y.cpu().numpy())
            
            del batch_data
    
    return np.array(all_preds), np.array(all_labels)


def compute_metrics(preds, labels, threshold=0.5):
    pred_labels = (preds >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(labels, pred_labels).ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    try:
        roc_auc = roc_auc_score(labels, preds)
    except:
        roc_auc = 0.5
    
    try:
        avg_precision = average_precision_score(labels, preds)
    except:
        avg_precision = 0
    
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "avg_precision": float(avg_precision),
    }


def plot_roc_curves(results, output_dir):
    plt.figure(figsize=(10, 8))
    
    colors = ['#667eea', '#f5576c', '#28a745', '#ffc107']
    
    for i, (model_name, preds, labels) in enumerate(results):
        fpr, tpr, _ = roc_curve(labels, preds)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
               label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    path = os.path.join(output_dir, "roc_auc.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_precision_recall_curves(results, output_dir):
    plt.figure(figsize=(10, 8))
    
    colors = ['#667eea', '#f5576c', '#28a745', '#ffc107']
    
    for i, (model_name, preds, labels) in enumerate(results):
        precision, recall, _ = precision_recall_curve(labels, preds)
        avg_precision = average_precision_score(labels, preds)
        
        plt.plot(recall, precision, color=colors[i % len(colors)], lw=2,
               label=f'{model_name} (AP = {avg_precision:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    path = os.path.join(output_dir, "precision_recall.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(cm_dict, output_dir, threshold=0.5):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, (model_name, cm) in enumerate(cm_dict.items()):
        ax = axes[idx]
        
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax, shrink=0.6)
        
        ax.set(xticks=[0, 1], yticks=[0, 1],
              xticklabels=['Pred: Legit', 'Pred: Fraud'],
              yticklabels=['Actual: Legit', 'Actual: Fraud'])
        
        thresh = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black", fontsize=14)
        
        ax.set_ylabel('True Label', fontsize=11)
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_title(f'{model_name}\nThreshold: {threshold}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_threshold_analysis(results, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, (model_name, preds, labels) in enumerate(results):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        thresholds = np.arange(0.1, 1.0, 0.05)
        precisions = []
        recalls = []
        f1s = []
        
        for t in thresholds:
            metrics = compute_metrics(preds, labels, t)
            precisions.append(metrics["precision"])
            recalls.append(metrics["recall"])
            f1s.append(metrics["f1"])
        
        ax.plot(thresholds, precisions, 'b-', lw=2, label='Precision')
        ax.plot(thresholds, recalls, 'r-', lw=2, label='Recall')
        ax.plot(thresholds, f1s, 'g-', lw=2, label='F1 Score')
        
        ax.set_xlabel('Threshold', fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(f'{model_name} - Threshold Analysis', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.1, 0.9])
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    path = os.path.join(output_dir, "threshold_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_fraud_probability_distribution(results, output_dir, threshold=0.5):
    plt.figure(figsize=(10, 6))
    
    colors = ['#667eea', '#f5576c']
    
    for i, (model_name, preds, labels) in enumerate(results):
        legit_mask = labels == 0
        fraud_mask = labels == 1
        
        plt.hist(preds[legit_mask], bins=50, alpha=0.6, color=colors[0],
                label=f'Legit (n={legit_mask.sum()})', density=True)
        plt.hist(preds[fraud_mask], bins=50, alpha=0.6, color=colors[1],
                label=f'Frauds (n={fraud_mask.sum()})', density=True)
        
        plt.axvline(threshold, color='red', linestyle='--', lw=2, label=f'Threshold ({threshold})')
    
    plt.xlabel('Fraud Probability', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Fraud Probability Distribution by True Label', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    path = os.path.join(output_dir, "probability_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def create_summary_table(results, metrics_by_model, threshold, output_dir):
    model_names = [r[0] for r in results]
    
    table_data = []
    for model_name in model_names:
        m = metrics_by_model[model_name]
        table_data.append([
            model_name,
            f'{m["roc_auc"]:.4f}',
            f'{m["avg_precision"]:.4f}',
            f'{m["precision"]:.4f}',
            f'{m["recall"]:.4f}',
            f'{m["specificity"]:.4f}',
            f'{m["f1"]:.4f}',
            f'{m["tp"]}/{m["tn"]}/{m["fp"]}/{m["fn"]}',
        ])
    
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')
    
    column_labels = ['Model', 'ROC AUC', 'Avg Prec', 'Precision', 'Recall', 'Specificity', 'F1', 'TP/TN/FP/FN']
    table = ax.table(cellText=table_data, colLabels=column_labels,
                  loc='center', cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    for i in range(len(column_labels)):
        table[(0, i)].set_facecolor('#667eea')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax.set_title(f'Model Performance Summary (Threshold: {threshold})',
               fontsize=14, fontweight='bold', pad=20)
    
    path = os.path.join(output_dir, "summary_table.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate ML Evaluation Figures")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Classification threshold")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for inference")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  ML Evaluation Figures Generator")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Device: {device}")
    
    print(f"\n[1/5] Loading graph from {GRAPH_PATH}...")
    data = torch.load(GRAPH_PATH, map_location=device, weights_only=False)
    
    tx_feature_dim = data["transaction"].x.size(1)
    num_tx = data["transaction"].x.size(0)
    labels = data["transaction"].y.cpu().numpy()
    print(f"  Total transactions: {num_tx:,}")
    print(f"  Feature dim: {tx_feature_dim}")
    print(f"  Fraud rate: {labels.mean():.4%}")
    
    print(f"\n[2/5] Creating temporal train/val split ({TRAIN_RATIO*100:.0f}%/{(1-TRAIN_RATIO)*100:.0f}%)...")
    num_train = int(num_tx * TRAIN_RATIO)
    val_idx = torch.arange(num_train, num_tx, device=device)
    print(f"  Validation samples: {len(val_idx):,}")
    
    results = []
    metrics_by_model = {}
    
    if os.path.exists(HETEROCONV_MODEL_PATH):
        print(f"\n[3/5] Loading HeteroConv model...")
        checkpoint = torch.load(HETEROCONV_MODEL_PATH, map_location=device, weights_only=False)
        
        model = HeteroConvFraudDetector(
            tx_feature_dim=tx_feature_dim,
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
        
        val_auc = checkpoint.get("val_auc", 0)
        print(f"  Model val_auc from checkpoint: {val_auc:.4f}")
        
        print(f"  Running inference...")
        preds, val_labels = run_inference(model, data, val_idx, device, args.batch_size)
        
        metrics = compute_metrics(preds, val_labels, args.threshold)
        metrics_by_model["HeteroConv"] = metrics
        
        results.append(("HeteroConv", preds, val_labels))
        
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
        
        del model
    
    if os.path.exists(BASELINE_MODEL_PATH):
        print(f"\n[4/5] Loading Baseline model...")
        try:
            checkpoint = torch.load(BASELINE_MODEL_PATH, map_location=device, weights_only=False)
            
            model = BaselineFraudDetector(
                tx_feature_dim=tx_feature_dim,
                hidden_dim=HIDDEN_DIM,
                num_layers=NUM_LAYERS,
                num_heads=NUM_HEADS,
                dropout=DROPOUT,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(device)
            
            val_auc = checkpoint.get("val_auc", 0)
            print(f"  Model val_auc from checkpoint: {val_auc:.4f}")
            
            print(f"  Running inference...")
            preds, val_labels = run_inference(model, data, val_idx, device, args.batch_size)
            
            metrics = compute_metrics(preds, val_labels, args.threshold)
            metrics_by_model["Baseline"] = metrics
            
            results.append(("Baseline", preds, val_labels))
            
            print(f"  ROC AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
            
            del model
        except Exception as e:
            print(f"  Warning: Could not load Baseline model: {e}")
            print(f"  Skipping Baseline model evaluation.")
    
    if not results:
        print("Error: No models found to evaluate!")
        sys.exit(1)
    
    print(f"\n[5/5] Generating plots...")
    
    plot_roc_curves(results, OUTPUT_DIR)
    plot_precision_recall_curves(results, OUTPUT_DIR)
    
    cm_dict = {}
    for model_name, preds, labels in results:
        pred_labels = (preds >= args.threshold).astype(int)
        cm = confusion_matrix(labels, pred_labels)
        cm_dict[model_name] = cm
    plot_confusion_matrix(cm_dict, OUTPUT_DIR, args.threshold)
    
    plot_threshold_analysis(results, OUTPUT_DIR)
    plot_fraud_probability_distribution(results, OUTPUT_DIR, args.threshold)
    create_summary_table(results, metrics_by_model, args.threshold, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("  Evaluation Complete!")
    print("=" * 60)
    
    for model_name, m in metrics_by_model.items():
        print(f"\n{model_name}:")
        print(f"  ROC AUC:        {m['roc_auc']:.4f}")
        print(f"  Avg Precision: {m['avg_precision']:.4f}")
        print(f"  Precision:    {m['precision']:.4f}")
        print(f"  Recall:      {m['recall']:.4f}")
        print(f"  F1 Score:    {m['f1']:.4f}")
    
    print(f"\nOutput files in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()