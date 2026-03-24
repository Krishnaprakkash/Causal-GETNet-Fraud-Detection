#!/usr/bin/env python3
"""
Configuration file for PJT2 Fraud Detection Training with GATConv.

Only includes parameters that should be tweaked between runs.
All other training parameters are hardcoded in train_gatconv.py.

Usage:
    from config_gatconv import MODEL_CONFIG
"""

# =============================================================================
# Model & Training Parameters (tweakable between runs)
# =============================================================================

MODEL_CONFIG = {
    # Hidden dimension for GNN layers (64-256)
    "hidden_dim": 128,

    # Number of GNN message passing layers (2-4)
    "num_layers": 2,

    # Number of attention heads for GATConv (4-8)
    "num_heads": 8,

    # Dropout rate (0.1-0.4)
    "dropout": 0.3,

    # Entity embedding bucket size (1000-100000)
    # Higher = less hash collisions but more memory
    "num_entity_nodes": 20000,

    # Mini-batch size (512-4096)
    # Smaller = better gradients, less memory per step
    "batch_size": 1024,
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
    },

    "low_memory": {
        "hidden_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.3,
        "num_entity_nodes": 10000,
        "batch_size": 4096,
    },

    "high_capacity": {
        "hidden_dim": 128,
        "num_layers": 3,
        "num_heads": 8,
        "dropout": 0.2,
        "num_entity_nodes": 100000,
        "batch_size": 1024,
    },
}


def apply_preset(preset_name: str) -> None:
    """Apply a preset configuration."""
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")
    MODEL_CONFIG.update(PRESETS[preset_name])
