#!/usr/bin/env python3
"""
Baseline Model Evaluation Script

Evaluates the baseline TransformerConv model (best_model.pt) on the validation set.
Generates evaluation metrics and plots.

Input:  data/processed/best_model.pt
        data/processed/hetero_graph_with_env.pt
Output: data/processed/figures/baseline_*.png
        Console metrics
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
)
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_transconv import MODEL_CONFIG

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(PROCESSED_DIR, "figures")

GRAPH_PATH = os.path.join(PROCESSED_DIR, "hetero_graph_with_env.pt")
MODEL_PATH = os.path.join(PROCESSED_DIR, "best_model.pt")

HIDDEN_DIM = MODEL_CONFIG["hidden_dim"]
NUM_LAYERS = MODEL_CONFIG["num_layers"]
NUM_HEADS = MODEL_CONFIG["num_heads"]
DROPOUT = MODEL_CONFIG["dropout"]
NUM_ENTITY_NODES = MODEL_CONFIG["num_entity_nodes"]
BATCH_SIZE = MODEL_CONFIG["batch_size"]

ENTITY_TYPES = [
    "card1", "card2", "card3", "card4", "card5", "card6",
    "ProductCD", "P_emaildomain", "addr1", "addr2", "dist1",
]

TRAIN_RATIO = 0.8


def get_subgraph_for_batch(data, batch_tx_indices, device):
    """Extract subgraph for a batch of transactions."""
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
    """Run inference on validation set."""
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
    """Compute evaluation metrics."""
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


def plot_roc_curve(preds, labels, output_dir, model_name="baseline"):
    """Plot ROC curve."""
    plt.figure(figsize=(10, 8))
    
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='#667eea', lw=2,
             label=f'{model_name} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Baseline Model', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    path = os.path.join(output_dir, f"{model_name}_roc_auc.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    
    return roc_auc


def plot_precision_recall_curve(preds, labels, output_dir, model_name="baseline"):
    """Plot Precision-Recall curve."""
    plt.figure(figsize=(10, 8))
    
    precision, recall, _ = precision_recall_curve(labels, preds)
    avg_precision = average_precision_score(labels, preds)
    
    plt.plot(recall, precision, color='#f5576c', lw=2,
             label=f'{model_name} (AP = {avg_precision:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve - Baseline Model', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    path = os.path.join(output_dir, f"{model_name}_precision_recall.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(metrics, output_dir, threshold, model_name="baseline"):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cm = np.array([[metrics['tn'], metrics['fp']], [metrics['fn'], metrics['tp']]])
    
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
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=16)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix (threshold={threshold})', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Baseline Model")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--threshold", type=float, default=0.8, help="Classification threshold")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for inference")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Baseline Model Evaluation")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Device: {device}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"\nError: Model not found at {MODEL_PATH}")
        print("Please train the baseline model first.")
        sys.exit(1)
    
    print(f"\n[1/4] Loading graph from {GRAPH_PATH}...")
    data = torch.load(GRAPH_PATH, map_location=device, weights_only=False)
    
    tx_feature_dim = data["transaction"].x.size(1)
    num_tx = data["transaction"].x.size(0)
    labels = data["transaction"].y.cpu().numpy()
    print(f"  Total transactions: {num_tx:,}")
    print(f"  Feature dim: {tx_feature_dim}")
    print(f"  Fraud rate: {labels.mean():.4%}")
    
    print(f"\n[2/4] Creating temporal train/val split ({TRAIN_RATIO*100:.0f}%/{(1-TRAIN_RATIO)*100:.0f}%)...")
    num_train = int(num_tx * TRAIN_RATIO)
    val_idx = torch.arange(num_train, num_tx, device=device)
    print(f"  Validation samples: {len(val_idx):,}")
    
    print(f"\n[3/4] Loading and evaluating Baseline model...")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    
    from train_transconv import FraudDetector
    
    model = FraudDetector(
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
    
    print(f"\n  Evaluation Metrics:")
    print(f"    ROC AUC:         {metrics['roc_auc']:.4f}")
    print(f"    Avg Precision:  {metrics['avg_precision']:.4f}")
    print(f"    F1 Score:       {metrics['f1']:.4f}")
    print(f"    Precision:       {metrics['precision']:.4f}")
    print(f"    Recall:         {metrics['recall']:.4f}")
    print(f"    Specificity:     {metrics['specificity']:.4f}")
    print(f"    Threshold:      {args.threshold}")
    print(f"\n    True Positives:  {metrics['tp']:,}")
    print(f"    True Negatives: {metrics['tn']:,}")
    print(f"    False Positives: {metrics['fp']:,}")
    print(f"    False Negatives: {metrics['fn']:,}")
    
    del model
    
    print(f"\n[4/4] Generating plots...")
    plot_roc_curve(preds, val_labels, OUTPUT_DIR, "baseline")
    plot_precision_recall_curve(preds, val_labels, OUTPUT_DIR, "baseline")
    plot_confusion_matrix(metrics, OUTPUT_DIR, args.threshold, "baseline")
    
    print("\n" + "=" * 60)
    print("  Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
