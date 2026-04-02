#!/usr/bin/env python3
"""
Configuration file for PJT2 Fraud Detection Training with HeteroConv.

Only includes parameters that should be tweaked between runs.
All other training parameters are hardcoded in train_heteroconv.py.

Usage:
    from config_heteroconv import MODEL_CONFIG
"""

# =============================================================================
# Model & Training Parameters (tweakable between runs)
# =============================================================================

MODEL_CONFIG = {
    # Hidden dimension for GNN layers (64-256)
    "hidden_dim": 128,

    # Number of GNN message passing layers (2-4)
    "num_layers": 2,

    # Number of attention heads for TransformerConv (4-8)
    "num_heads": 8,

    # Dropout rate (0.1-0.4)
    "dropout": 0.3,

    # Entity embedding bucket size (1000-100000)
    # Higher = less hash collisions but more memory
    "num_entity_nodes": 20000,

    # Mini-batch size (512-4096)
    # Smaller = better gradients, less memory per step
    "batch_size": 1024,

    # HeteroConv-specific parameters
    # Number of attention heads for GATConv in HeteroConv layer
    "heteroconv_heads": 4,

    # Dropout rate for HeteroConv layer
    "heteroconv_dropout": 0.2,

    # Enable/disable HeteroConv layer
    # If False, falls back to baseline TransformerConv architecture
    "use_heteroconv": True,

    # IRM-specific parameters
    # Enable IRM training (default: False for backward compatibility)
    "use_irm": False,

    # Penalty weight (lambda) for IRM penalty
    # Range: 1e0 to 1e4, start with 1e2
    "irm_penalty_weight": 1e2,

    # Number of epochs to anneal penalty weight
    # Gradually increase penalty from 0 to target value
    "irm_penalty_anneal_epochs": 10,

    # Which environment to use for IRM
    # Options: "env_time", "env_region", "env_fraud_rate", or "all"
    "irm_environment": "env_time",
}


# =============================================================================
# Parameter Presets
# =============================================================================
PRESETS = {
    "optimized": {
        "hidden_dim": 96,
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.25,
        "num_entity_nodes": 50000,
        "batch_size": 2048,
        "heteroconv_heads": 4,
        "heteroconv_dropout": 0.2,
        "use_heteroconv": True,
    },

    "low_memory": {
        "hidden_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.3,
        "num_entity_nodes": 10000,
        "batch_size": 4096,
        "heteroconv_heads": 2,
        "heteroconv_dropout": 0.3,
        "use_heteroconv": True,
    },

    "high_capacity": {
        "hidden_dim": 128,
        "num_layers": 3,
        "num_heads": 8,
        "dropout": 0.2,
        "num_entity_nodes": 100000,
        "batch_size": 1024,
        "heteroconv_heads": 8,
        "heteroconv_dropout": 0.15,
        "use_heteroconv": True,
    },

    "baseline": {
        "hidden_dim": 128,
        "num_layers": 2,
        "num_heads": 8,
        "dropout": 0.3,
        "num_entity_nodes": 20000,
        "batch_size": 1024,
        "heteroconv_heads": 4,
        "heteroconv_dropout": 0.2,
        "use_heteroconv": False,  # Disable HeteroConv for baseline comparison
    },

    "irm_time": {
        "hidden_dim": 128,
        "num_layers": 2,
        "num_heads": 8,
        "dropout": 0.3,
        "num_entity_nodes": 20000,
        "batch_size": 1024,
        "heteroconv_heads": 4,
        "heteroconv_dropout": 0.2,
        "use_heteroconv": True,
        "use_irm": True,
        "irm_penalty_weight": 1e2,
        "irm_penalty_anneal_epochs": 10,
        "irm_environment": "env_time",
    },

    "irm_region": {
        "hidden_dim": 128,
        "num_layers": 2,
        "num_heads": 8,
        "dropout": 0.3,
        "num_entity_nodes": 20000,
        "batch_size": 1024,
        "heteroconv_heads": 4,
        "heteroconv_dropout": 0.2,
        "use_heteroconv": True,
        "use_irm": True,
        "irm_penalty_weight": 1e2,
        "irm_penalty_anneal_epochs": 10,
        "irm_environment": "env_region",
    },

    "irm_fraud_rate": {
        "hidden_dim": 128,
        "num_layers": 2,
        "num_heads": 8,
        "dropout": 0.3,
        "num_entity_nodes": 20000,
        "batch_size": 1024,
        "heteroconv_heads": 4,
        "heteroconv_dropout": 0.2,
        "use_heteroconv": True,
        "use_irm": True,
        "irm_penalty_weight": 1e2,
        "irm_penalty_anneal_epochs": 10,
        "irm_environment": "env_fraud_rate",
    },

    "irm_all": {
        "hidden_dim": 128,
        "num_layers": 2,
        "num_heads": 8,
        "dropout": 0.3,
        "num_entity_nodes": 20000,
        "batch_size": 1024,
        "heteroconv_heads": 4,
        "heteroconv_dropout": 0.2,
        "use_heteroconv": True,
        "use_irm": True,
        "irm_penalty_weight": 1e2,
        "irm_penalty_anneal_epochs": 10,
        "irm_environment": "all",
    },
}


def apply_preset(preset_name: str) -> None:
    """Apply a preset configuration."""
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")
    MODEL_CONFIG.update(PRESETS[preset_name])
