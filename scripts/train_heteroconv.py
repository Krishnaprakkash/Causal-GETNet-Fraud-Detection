#!/usr/bin/env python3
"""
HeteroConv Model Training for PJT2 Fraud Detection

Trains a Heterogeneous HeteroConv model on the transaction graph
with temporal train/validation split.

Architecture:
  Entity Embeddings → HeteroConv (GATConv per entity type) → TransformerConv (cross-entity) → Classifier

Input:  PJT2/data/processed/hetero_graph_with_env.pt
Output: PJT2/data/processed/best_model_heteroconv.pt

Configuration: All parameters are in scripts/config_heteroconv.py
"""

import os
import sys
import argparse
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, TransformerConv
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Import configuration from config.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_heteroconv import MODEL_CONFIG

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

GRAPH_PATH = os.path.join(PROCESSED_DIR, "hetero_graph_with_env.pt")
MODEL_OUTPUT_PATH = os.path.join(PROCESSED_DIR, "best_model_heteroconv.pt")

# Model hyperparameters (from config_heteroconv.py - tweakable)
HIDDEN_DIM = MODEL_CONFIG["hidden_dim"]
NUM_LAYERS = MODEL_CONFIG["num_layers"]
NUM_HEADS = MODEL_CONFIG["num_heads"]
DROPOUT = MODEL_CONFIG["dropout"]
NUM_ENTITY_NODES = MODEL_CONFIG["num_entity_nodes"]
BATCH_SIZE = MODEL_CONFIG["batch_size"]
HETEROCONV_HEADS = MODEL_CONFIG["heteroconv_heads"]
HETEROCONV_DROPOUT = MODEL_CONFIG["heteroconv_dropout"]
USE_HETEROCONV = MODEL_CONFIG["use_heteroconv"]

# Entity types (hardcoded - from build_graph.py ENTITY_COLS)
ENTITY_TYPES = [
    "card1", "card2", "card3", "card4", "card5", "card6",
    "ProductCD", "P_emaildomain", "addr1", "addr2", "dist1",
]

# Training hyperparameters (hardcoded - no need to tweak)
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
TRAIN_RATIO = 0.8
USE_MINIBATCH = True

# ---------------------------------------------------------------------------
# Model Definition
# ---------------------------------------------------------------------------


class HeteroConvFraudDetector(nn.Module):
    """
    Heterogeneous GNN for fraud detection using HeteroConv + TransformerConv.
    
    Architecture:
    - Entity embeddings: 11 types × nn.Embedding(20000, hidden_dim)
    - HeteroConv layer: GATConv for each entity type (type-specific learning)
    - TransformerConv layers: Cross-entity attention
    - Classifier MLP: concat(tx_features, entity_messages) → 128 → 64 → 1
    
    Key Innovation:
    - HeteroConv applies separate GATConv to each entity type
    - This enables type-specific feature learning
    - TransformerConv then handles cross-entity attention
    """

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

        # Entity embeddings (11 types × 20,000 nodes)
        self.entity_embeddings = nn.ModuleDict({
            entity_type: nn.Embedding(NUM_ENTITY_NODES, hidden_dim)
            for entity_type in self.entity_types
        })

        # Initial projection of transaction features
        self.tx_proj = nn.Linear(tx_feature_dim, hidden_dim)

        # HeteroConv layer - GATConv for each entity type
        if use_heteroconv:
            conv_dict = {}
            for entity_type in self.entity_types:
                conv_dict[(entity_type, "uses", "transaction")] = GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=heteroconv_heads,
                    concat=False,  # Output hidden_dim (not hidden_dim * heads)
                    dropout=heteroconv_dropout,
                    add_self_loops=False,  # Don't add self-loops for heterogeneous edges
                )
            self.heteroconv = HeteroConv(conv_dict, aggr='sum')
            
            # Layer norm after HeteroConv
            self.heteroconv_norm = nn.LayerNorm(hidden_dim)

        # TransformerConv layers for cross-entity attention
        self.convs = nn.ModuleList()
        for layer_idx in range(num_layers):
            # First layer: input dim = hidden_dim, output dim = hidden_dim
            # Subsequent layers: input dim = hidden_dim * 2 (from cat), output = hidden_dim
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

        # Batch norm for each entity type
        self.batch_norms = nn.ModuleDict({
            entity_type: nn.LayerNorm(hidden_dim * num_heads if num_layers == 1 else hidden_dim)
            for entity_type in self.entity_types
        })

        # Classifier MLP: concat(tx_features, aggregated entity messages) → 128 → 64 → 1
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
        """
        Forward pass through the heterogeneous graph.
        
        Args:
            data: HeteroData object with:
                - data['transaction'].x: [N_tx, tx_feature_dim]
                - data[entity_type].x: [N_entity, 1] (entity indices)
                - Edge types: (entity, 'uses', 'transaction')
        
        Returns:
            logits: [N_tx, 1] prediction logits
        """
        # Get transaction features
        tx_x = data["transaction"].x  # [N_tx, tx_feature_dim]

        # Project transaction features to hidden dimension
        tx_h = self.tx_proj(tx_x)  # [N_tx, hidden_dim]

        # Prepare node features for HeteroConv
        x_dict = {"transaction": tx_h}
        for entity_type in self.entity_types:
            entity_x = data[entity_type].x  # [N_entity_nodes, 1] - these are indices
            entity_emb = self.entity_embeddings[entity_type](entity_x.squeeze(-1).long())  # [N_entity, hidden_dim]
            x_dict[entity_type] = entity_emb

        # Get edge index dict
        edge_index_dict = {}
        for entity_type in self.entity_types:
            edge_type = (entity_type, "uses", "transaction")
            # Safely check if edge_type exists in data
            try:
                if edge_type in data.edge_index_dict:
                    edge_index_dict[edge_type] = data.edge_index_dict[edge_type]
            except (KeyError, AttributeError):
                # No edges in data, skip this edge type
                pass

        # Apply HeteroConv (GATConv per entity type) if enabled
        if self.use_heteroconv:
            out_dict = self.heteroconv(x_dict, edge_index_dict)
            # Apply layer norm
            for entity_type in self.entity_types:
                if entity_type in out_dict:
                    out_dict[entity_type] = self.heteroconv_norm(out_dict[entity_type])
        else:
            # Fallback: use entity embeddings directly
            out_dict = {entity_type: x_dict[entity_type] for entity_type in self.entity_types}

        # Process each entity type and aggregate messages
        entity_messages_list = []

        for entity_type in self.entity_types:
            # Get entity representations (from HeteroConv or embeddings)
            if entity_type in out_dict:
                h = out_dict[entity_type]
            else:
                # Fallback to embeddings if not in out_dict
                entity_x = data[entity_type].x
                h = self.entity_embeddings[entity_type](entity_x.squeeze(-1).long())

            # Get edge index for this entity type
            edge_type = (entity_type, "uses", "transaction")
            if edge_type in edge_index_dict:
                edge_index = edge_index_dict[edge_type]

                # Apply TransformerConv for cross-entity attention
                conv = self.convs[0]
                h = conv(h, edge_index)  # [N_entity, hidden_dim * num_heads or hidden_dim]

                # Apply batch norm
                h = self.batch_norms[entity_type](h)

                # For transaction nodes, we need to aggregate messages from entities
                # Create a mapping from transaction node index to message
                num_tx = tx_x.size(0)
                agg_h = torch.zeros(num_tx, h.size(-1), device=h.device)

                # Aggregate: sum messages for each transaction
                src, dst = edge_index
                agg_h.index_add_(0, dst, h[src])

                entity_messages_list.append(agg_h)

        # Concatenate all entity messages
        if entity_messages_list:
            entity_messages = torch.cat(entity_messages_list, dim=-1)  # [N_tx, hidden_dim * num_entity_types]
        else:
            entity_messages = torch.zeros(tx_x.size(0), self.hidden_dim * self.num_entity_types, device=tx_x.device)

        # Concatenate transaction features with entity messages
        combined = torch.cat([tx_x, entity_messages], dim=-1)  # [N_tx, tx_feature_dim + hidden_dim * num_entity_types]

        # Classify
        logits = self.classifier(combined)  # [N_tx, 1]

        return logits


# ---------------------------------------------------------------------------
# Training Functions
# ---------------------------------------------------------------------------


def compute_pos_weight(labels: torch.Tensor) -> float:
    """Compute positive weight for BCE loss based on class imbalance."""
    num_pos = labels.sum().item()
    num_neg = len(labels) - num_pos
    return num_neg / max(num_pos, 1)


def get_subgraph_for_batch(data, batch_tx_indices, device):
    """
    Extract a subgraph containing only the batch transactions and their neighbors.
    This reduces memory usage by loading ONLY entity nodes connected to the batch.
    """
    from torch_geometric.data import HeteroData
    
    # Create subgraph
    subgraph = HeteroData()
    
    # Get transaction features and labels for batch
    subgraph['transaction'].x = data['transaction'].x[batch_tx_indices].to(device)
    subgraph['transaction'].y = data['transaction'].y[batch_tx_indices].to(device)
    
    # Get edge indices for each entity type
    batch_tx_set = set(batch_tx_indices.cpu().tolist())
    batch_tx_indices_cpu = batch_tx_indices.cpu()
    
    # Collect ONLY entity nodes that have edges to batch transactions
    for entity_type in ENTITY_TYPES:
        edge_type = (entity_type, "uses", "transaction")
        # Safely check if edge_type exists in data
        try:
            has_edges = edge_type in data.edge_index_dict
        except (KeyError, AttributeError):
            has_edges = False
        
        if not has_edges:
            # No edges for this entity type - add empty x with shape [1, 1] for model compatibility
            subgraph[entity_type].x = torch.tensor([[0]], dtype=torch.long, device=device)
            continue
            
        edge_index = data.edge_index_dict[edge_type]
        src = edge_index[0].cpu()
        dst = edge_index[1].cpu()
        
        # Filter to edges where destination is in batch
        mask = torch.tensor([dst_i in batch_tx_set for dst_i in dst], dtype=torch.bool)
        
        if mask.sum() > 0:
            # Get unique entity node indices connected to batch transactions
            connected_src = src[mask]
            connected_entity_indices = connected_src.unique().tolist()
            
            # Create remapping: original index -> 0..N-1
            entity_idx_to_new = {idx: new_idx for new_idx, idx in enumerate(connected_entity_indices)}
            
            # Add entity nodes with remapped indices for embedding lookup
            new_entity_indices = torch.tensor(connected_entity_indices, dtype=torch.long, device=device).unsqueeze(-1)
            subgraph[entity_type].x = new_entity_indices
            
            # Remap edge indices: entity nodes to 0..N-1, transactions to 0..batch_size-1
            tx_idx_to_batch = {idx: i for i, idx in enumerate(batch_tx_indices_cpu)}
            
            filtered_edges = edge_index[:, mask]
            original_src = filtered_edges[0].cpu()
            original_dst = filtered_edges[1].cpu()
            
            new_src = torch.tensor([entity_idx_to_new[s.item()] for s in original_src], dtype=torch.long, device=device)
            new_dst = torch.tensor([tx_idx_to_batch[dst_i.item()] for dst_i in original_dst], dtype=torch.long, device=device)
            
            subgraph[edge_type].edge_index = torch.stack([new_src, new_dst])
        else:
            # No connected entity nodes - add dummy entity for model compatibility
            subgraph[entity_type].x = torch.tensor([[0]], dtype=torch.long, device=device)
    
    return subgraph


def train_epoch_minibatch(model, data, train_idx, optimizer, criterion, device, batch_size):
    """Train for one epoch using mini-batches."""
    model.train()
    
    # Shuffle training indices
    num_train = len(train_idx)
    perm = torch.randperm(num_train, device=device)
    train_idx_shuffled = train_idx[perm]
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    num_batches = (num_train + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_train)
        batch_tx_indices = train_idx_shuffled[start:end]
        
        optimizer.zero_grad()
        
        # Get subgraph for this batch
        batch_data = get_subgraph_for_batch(data, batch_tx_indices, device)
        
        # Forward pass
        logits = model(batch_data)
        logits = logits.squeeze(-1)
        
        # Compute loss
        labels = batch_data["transaction"].y
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(batch_tx_indices)
        all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Memory cleanup after each batch
        del batch_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Compute metrics
    avg_loss = total_loss / num_train
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    try:
        train_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        train_auc = 0.5
    
    return avg_loss, train_auc


def train_epoch_minibatch_irm(model, data, train_idx, optimizer, criterion, device, batch_size, env_labels, penalty_weight):
    """Train for one epoch using mini-batches with IRM loss."""
    model.train()
    
    # Shuffle training indices
    num_train = len(train_idx)
    perm = torch.randperm(num_train, device=device)
    train_idx_shuffled = train_idx[perm]
    
    total_loss = 0.0
    total_erm_loss = 0.0
    total_penalty = 0.0
    all_preds = []
    all_labels = []
    
    num_batches = (num_train + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_train)
        batch_tx_indices = train_idx_shuffled[start:end]
        
        optimizer.zero_grad()
        
        # Get subgraph for this batch
        batch_data = get_subgraph_for_batch(data, batch_tx_indices, device)
        
        # Forward pass
        logits = model(batch_data)
        logits = logits.squeeze(-1)
        
        # Get labels and environment labels for this batch
        labels = batch_data["transaction"].y
        batch_env_labels = env_labels[batch_tx_indices]
        
        # Compute IRM loss
        loss, erm_loss, penalty = compute_irm_loss_per_env(
            logits, labels, batch_env_labels, penalty_weight
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(batch_tx_indices)
        total_erm_loss += erm_loss * len(batch_tx_indices)
        total_penalty += penalty * len(batch_tx_indices)
        all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Memory cleanup
        del batch_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Compute metrics
    avg_loss = total_loss / num_train
    avg_erm_loss = total_erm_loss / num_train
    avg_penalty = total_penalty / num_train
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    try:
        train_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        train_auc = 0.5
    
    return avg_loss, train_auc, avg_erm_loss, avg_penalty


def evaluate_minibatch(model, data, val_idx, criterion, device, batch_size):
    """Evaluate on validation set using mini-batches."""
    model.eval()
    
    num_val = len(val_idx)
    num_batches = (num_val + batch_size - 1) // batch_size
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, num_val)
            batch_tx_indices = val_idx[start:end]
            
            # Get subgraph for this batch
            batch_data = get_subgraph_for_batch(data, batch_tx_indices, device)
            
            # Forward pass
            logits = model(batch_data)
            logits = logits.squeeze(-1)
            
            # Compute loss
            labels = batch_data["transaction"].y
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * len(batch_tx_indices)
            all_preds.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Memory cleanup after each batch
            del batch_data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Compute metrics
    avg_loss = total_loss / num_val
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    try:
        val_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        val_auc = 0.5
    
    return avg_loss, val_auc


def train_epoch(model, data, train_idx, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()

    # Forward pass
    logits = model(data)
    logits = logits.squeeze(-1)

    # Compute loss on training samples only
    loss = criterion(logits[train_idx], data["transaction"].y[train_idx])

    # Backward pass
    loss.backward()
    optimizer.step()

    # Compute training metrics
    with torch.no_grad():
        preds = torch.sigmoid(logits[train_idx])
        labels = data["transaction"].y[train_idx].cpu().numpy()
        preds_np = preds.cpu().numpy()

        train_loss = F.binary_cross_entropy_with_logits(
            logits[train_idx], data["transaction"].y[train_idx]
        ).item()

        try:
            train_auc = roc_auc_score(labels, preds_np)
        except ValueError:
            train_auc = 0.5  # Handle case with only one class

    return train_loss, train_auc


def train_epoch_irm(model, data, train_idx, optimizer, criterion, device, env_labels, penalty_weight):
    """Train for one epoch with IRM loss (full graph)."""
    model.train()
    optimizer.zero_grad()

    # Forward pass
    logits = model(data)
    logits = logits.squeeze(-1)

    # Get labels and environment labels for training samples
    labels = data["transaction"].y[train_idx]
    train_env_labels = env_labels[train_idx]
    train_logits = logits[train_idx]
    
    # Compute IRM loss
    loss, erm_loss, penalty = compute_irm_loss_per_env(
        train_logits, labels, train_env_labels, penalty_weight
    )

    # Backward pass
    loss.backward()
    optimizer.step()

    # Compute training metrics
    with torch.no_grad():
        preds = torch.sigmoid(train_logits)
        labels_np = labels.cpu().numpy()
        preds_np = preds.cpu().numpy()

        train_loss = F.binary_cross_entropy_with_logits(
            train_logits, labels
        ).item()

        try:
            train_auc = roc_auc_score(labels_np, preds_np)
        except ValueError:
            train_auc = 0.5  # Handle case with only one class

    return train_loss, train_auc, erm_loss, penalty


def evaluate(model, data, val_idx, criterion, device):
    """Evaluate on validation set."""
    model.eval()

    with torch.no_grad():
        logits = model(data)
        logits = logits.squeeze(-1)

        val_loss = F.binary_cross_entropy_with_logits(
            logits[val_idx], data["transaction"].y[val_idx]
        ).item()

        preds = torch.sigmoid(logits[val_idx])
        labels = data["transaction"].y[val_idx].cpu().numpy()
        preds_np = preds.cpu().numpy()

        try:
            val_auc = roc_auc_score(labels, preds_np)
        except ValueError:
            val_auc = 0.5

    return val_loss, val_auc


# ---------------------------------------------------------------------------
# IRM Loss Functions
# ---------------------------------------------------------------------------


def get_environment_labels(data, env_type):
    """
    Get environment labels based on environment type.
    
    Args:
        data: HeteroData object
        env_type: "env_time", "env_region", "env_fraud_rate", or "all"
    
    Returns:
        env_labels: [N] tensor of environment IDs
    """
    if env_type == "env_time":
        return data["transaction"].env_time
    elif env_type == "env_region":
        return data["transaction"].env_region
    elif env_type == "env_fraud_rate":
        return data["transaction"].env_fraud_rate
    elif env_type == "all":
        # Combine all environments: create unique ID for each combination
        # env_time: 0-7, env_region: 0-4, env_fraud_rate: 0-2
        # Combined: env_time * 15 + env_region * 3 + env_fraud_rate
        env_time = data["transaction"].env_time
        env_region = data["transaction"].env_region
        env_fraud_rate = data["transaction"].env_fraud_rate
        
        combined = env_time * 15 + env_region * 3 + env_fraud_rate
        return combined
    else:
        raise ValueError(f"Unknown environment type: {env_type}")


def compute_irm_loss_per_env(logits, labels, env_labels, penalty_weight):
    """
    Compute IRMv1 loss with per-environment gradient penalty.
    
    The penalty encourages the same optimal classifier across environments.
    
    Args:
        logits: [N] prediction logits
        labels: [N] binary labels
        env_labels: [N] environment IDs
        penalty_weight: λ for IRM penalty
    
    Returns:
        total_loss: L_ERM + λ * L_penalty
        erm_loss: ERM loss (for logging)
        penalty: IRM penalty (for logging)
    """
    # Compute ERM loss (average across all samples)
    erm_loss = F.binary_cross_entropy_with_logits(logits, labels)
    
    # Get unique environments
    unique_envs = torch.unique(env_labels)
    
    # Compute per-environment risk and penalty
    penalties = []
    for env_id in unique_envs:
        env_mask = env_labels == env_id
        if env_mask.sum() == 0:
            continue
        
        env_logits = logits[env_mask]
        env_labels_env = labels[env_mask]
        
        # Compute gradient of loss w.r.t. dummy scalar w
        # We want gradient w.r.t. w evaluated at w=1
        w = torch.ones(1, device=logits.device, requires_grad=True)
        scaled_logits = w * env_logits
        
        env_loss = F.binary_cross_entropy_with_logits(scaled_logits, env_labels_env)
        grad_w = torch.autograd.grad(env_loss, w, create_graph=True)[0]
        
        penalties.append(grad_w.pow(2))
    
    # Average penalty across environments
    if penalties:
        penalty = torch.stack(penalties).mean()
    else:
        penalty = torch.tensor(0.0, device=logits.device)
    
    # Total loss
    total_loss = erm_loss + penalty_weight * penalty
    
    return total_loss, erm_loss.item(), penalty.item()


# ---------------------------------------------------------------------------
# Main Training Pipeline
# ---------------------------------------------------------------------------


def main():
    # Ensure output directory exists early
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="PJT2 HeteroConv Model Training")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training (useful when GPU memory is insufficient)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Mini-batch size for training")
    parser.add_argument("--full-graph", action="store_true", help="Use full graph instead of mini-batching")
    parser.add_argument("--preset", type=str, default="optimized", 
                        choices=["optimized", "low_memory", "high_capacity", "baseline",
                                 "irm_time", "irm_region", "irm_fraud_rate", "irm_all"],
                        help="Configuration preset to use")
    
    # IRM-specific arguments
    parser.add_argument("--irm", action="store_true", help="Enable IRM training")
    parser.add_argument("--irm-penalty-weight", type=float, default=MODEL_CONFIG["irm_penalty_weight"],
                        help="IRM penalty weight (lambda)")
    parser.add_argument("--irm-env", type=str, default=MODEL_CONFIG["irm_environment"],
                        choices=["env_time", "env_region", "env_fraud_rate", "all"],
                        help="Environment type for IRM")
    parser.add_argument("--irm-anneal-epochs", type=int, default=MODEL_CONFIG["irm_penalty_anneal_epochs"],
                        help="Epochs to anneal IRM penalty weight")
    
    args = parser.parse_args()

    # Print configuration
    print("=" * 60)
    print("PJT2 HeteroConv Model Training")
    print("=" * 60)
    print(f"Model: hidden_dim={HIDDEN_DIM}, layers={NUM_LAYERS}, heads={NUM_HEADS}, dropout={DROPOUT}")
    print(f"Entity nodes: {NUM_ENTITY_NODES:,}, Batch size: {BATCH_SIZE}")
    print(f"HeteroConv: enabled={USE_HETEROCONV}, heads={HETEROCONV_HEADS}, dropout={HETEROCONV_DROPOUT}")
    print("=" * 60)

    # Set device based on availability and command-line flag
    if args.cpu:
        device = torch.device("cpu")
        print("Device: CPU (forced via --cpu flag)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        # Show GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Device: {device} ({gpu_name})")
        print(f"GPU Memory Total: {gpu_memory_total:.2f} GB")
        
        # Show current memory usage
        torch.cuda.reset_peak_memory_stats()
        mem_allocated = torch.cuda.memory_allocated(0) / 1e9
        mem_reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"GPU Memory Allocated: {mem_allocated:.2f} GB")
        print(f"GPU Memory Reserved: {mem_reserved:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Device: CPU (CUDA not available)")

    # Mini-batch settings
    use_minibatch = not args.full_graph
    batch_size = args.batch_size
    
    print(f"\nMini-batch mode: {use_minibatch}")
    print(f"Batch size: {batch_size}")

    # Load graph
    print(f"\n[1/5] Loading graph from {GRAPH_PATH}...")
    data = torch.load(GRAPH_PATH, map_location=device, weights_only=False)
    print(f"      Loaded graph with {data['transaction'].x.size(0)} transactions")
    print(f"      Transaction feature dim: {data['transaction'].x.size(1)}")

    # Get feature dimension
    tx_feature_dim = data["transaction"].x.size(1)
    num_tx = data["transaction"].x.size(0)

    # Compute pos_weight for class imbalance
    labels = data["transaction"].y
    pos_weight = compute_pos_weight(labels)
    print(f"      Fraud rate: {labels.mean().item():.4%}")
    print(f"      Pos weight: {pos_weight:.2f}")

    # Temporal split (graph is already sorted by TransactionDT)
    print(f"\n[2/5] Creating temporal train/val split ({TRAIN_RATIO*100:.0f}%/{(1-TRAIN_RATIO)*100:.0f}%)...")
    num_train = int(num_tx * TRAIN_RATIO)
    train_idx = torch.arange(num_train, device=device)
    val_idx = torch.arange(num_train, num_tx, device=device)
    print(f"      Train: {len(train_idx)} transactions")
    print(f"      Val:   {len(val_idx)} transactions")

    # IRM setup
    use_irm = args.irm or args.preset.startswith("irm_")
    irm_penalty_weight = args.irm_penalty_weight
    irm_env_type = args.irm_env
    irm_anneal_epochs = args.irm_anneal_epochs
    
    if use_irm:
        print(f"\nIRM Training Enabled:")
        print(f"  Environment: {irm_env_type}")
        print(f"  Penalty weight: {irm_penalty_weight}")
        print(f"  Anneal epochs: {irm_anneal_epochs}")
        
        # Get environment labels
        env_labels = get_environment_labels(data, irm_env_type)
        print(f"  Environment distribution:")
        unique_envs = torch.unique(env_labels)
        for env_id in unique_envs:
            count = (env_labels == env_id).sum().item()
            print(f"    Env {env_id}: {count:,} transactions")
    else:
        env_labels = None

    # Create model
    print(f"\n[3/5] Creating HeteroConvFraudDetector model...")
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
    model = model.to(device)
    print(f"      Hidden dim: {HIDDEN_DIM}")
    print(f"      Num layers: {NUM_LAYERS}")
    print(f"      Num heads: {NUM_HEADS}")
    print(f"      Entity bucket size: {NUM_ENTITY_NODES:,}")
    print(f"      HeteroConv enabled: {USE_HETEROCONV}")
    print(f"      HeteroConv heads: {HETEROCONV_HEADS}")
    print(f"      Mini-batch enabled: {use_minibatch}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Trainable parameters: {num_params:,}")

    # Setup optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    # Training loop
    print(f"\n[4/5] Training for {NUM_EPOCHS} epochs...")
    if use_minibatch:
        print(f"      Using mini-batch training (batch_size={batch_size})")
    else:
        print("      Using full graph training")
    print("-" * 60)

    best_val_auc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_train_auc = 0.0
    best_train_loss = 0.0

    for epoch in range(NUM_EPOCHS):
        try:
            # Anneal penalty weight for IRM
            if use_irm and epoch < irm_anneal_epochs:
                current_penalty = irm_penalty_weight * (epoch / irm_anneal_epochs)
            else:
                current_penalty = irm_penalty_weight if use_irm else 0.0
            
            # Train
            if use_minibatch:
                if use_irm:
                    train_loss, train_auc, erm_loss, penalty = train_epoch_minibatch_irm(
                        model, data, train_idx, optimizer, criterion, device, batch_size,
                        env_labels, current_penalty
                    )
                else:
                    train_loss, train_auc = train_epoch_minibatch(
                        model, data, train_idx, optimizer, criterion, device, batch_size
                    )
                # Validate
                val_loss, val_auc = evaluate_minibatch(
                    model, data, val_idx, criterion, device, batch_size
                )
            else:
                if use_irm:
                    train_loss, train_auc, erm_loss, penalty = train_epoch_irm(
                        model, data, train_idx, optimizer, criterion, device,
                        env_labels, current_penalty
                    )
                else:
                    train_loss, train_auc = train_epoch(
                        model, data, train_idx, optimizer, criterion, device
                    )
                # Validate
                val_loss, val_auc = evaluate(
                    model, data, val_idx, criterion, device
                )
        except Exception as e:
            print(f"\nERROR at epoch {epoch+1}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            print("Stopping training due to error.")
            break

        # Print progress
        if use_irm:
            print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f}, AUC: {train_auc:.4f} | "
                  f"ERM Loss: {erm_loss:.4f}, Penalty: {penalty:.4f} | "
                  f"Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")
        else:
            print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f}, AUC: {train_auc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

        # Early stopping check
        print(f"           DEBUG: val_auc={val_auc:.4f}, best_val_auc={best_val_auc:.4f}, epochs_without_improvement={epochs_without_improvement}")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            # Save best model (with train metrics too)
            best_train_auc = train_auc
            best_train_loss = train_loss
            # Ensure directory exists before saving
            os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "val_loss": val_loss,
                "train_auc": train_auc,
                "train_loss": train_loss,
            }, MODEL_OUTPUT_PATH)
            print(f"           -> Best model saved! (val_auc: {val_auc:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"           DEBUG: After increment, epochs_without_improvement={epochs_without_improvement}, PATIENCE={EARLY_STOPPING_PATIENCE}")
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {epochs_without_improvement} epochs)")
                break
        
        # Memory cleanup between epochs
        if device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            
            # Show memory stats occasionally
            if (epoch + 1) % 5 == 0:
                mem_allocated = torch.cuda.memory_allocated(0) / 1e9
                print(f"           [GPU Memory: {mem_allocated:.2f} GB allocated]")

    print("-" * 60)

    # Load best model and final evaluation
    print(f"\n[5/5] Loading best model from epoch {best_epoch}...")
    checkpoint = torch.load(MODEL_OUTPUT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Final evaluation
    if use_minibatch:
        final_val_loss, final_val_auc = evaluate_minibatch(model, data, val_idx, criterion, device, batch_size)
    else:
        final_val_loss, final_val_auc = evaluate(model, data, val_idx, criterion, device)

    # Final memory stats
    if device.type == "cuda":
        mem_allocated = torch.cuda.memory_allocated(0) / 1e9
        mem_reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"\nFinal GPU Memory: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best epoch: {best_epoch}")
    print(f"Best Train AUC: {best_train_auc:.4f}, Train Loss: {best_train_loss:.4f}")
    print(f"Best Val AUC: {final_val_auc:.4f}, Val Loss: {final_val_loss:.4f}")
    print(f"Model saved to: {MODEL_OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
