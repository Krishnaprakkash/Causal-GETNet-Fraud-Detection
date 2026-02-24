import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os

def visualize():

    data = torch.load(
        "../data/processed/graph_data.pt",
        weights_only=False
    )

    output_dir = "../data/processed/visualizecard1"
    os.makedirs(output_dir, exist_ok=True)

    labels = data["transaction"].y.numpy()
    edge_index = data["transaction", "uses", "card"].edge_index

    tx_nodes = edge_index[0].numpy()
    card_nodes = edge_index[1].numpy()

    # ------------------------------------------------
    # 1️⃣ Card Degree Distribution
    # ------------------------------------------------
    card_degree = Counter(card_nodes)
    card_degrees = list(card_degree.values())

    plt.figure()
    plt.hist(card_degrees, bins=50)
    plt.title("Card Degree Distribution")
    plt.xlabel("Transactions per Card")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/card_degree_distribution.png")
    plt.close()

    # ------------------------------------------------
    # 2️⃣ Fraud Ratio per Card
    # ------------------------------------------------
    fraud_per_card = {}

    for tx_idx, card_idx in zip(tx_nodes, card_nodes):
        fraud_per_card.setdefault(card_idx, []).append(labels[tx_idx])

    fraud_ratios = [
        np.mean(v) for v in fraud_per_card.values() if len(v) > 1
    ]

    plt.figure()
    plt.hist(fraud_ratios, bins=50)
    plt.title("Fraud Ratio per Card")
    plt.xlabel("Fraud Ratio")
    plt.ylabel("Number of Cards")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fraud_ratio_per_card.png")
    plt.close()

    # ------------------------------------------------
    # 3️⃣ Fraud vs Non-Fraud Transaction Degree
    # ------------------------------------------------
    tx_degree = Counter(tx_nodes)

    fraud_deg = []
    nonfraud_deg = []

    for tx_idx, deg in tx_degree.items():
        if labels[tx_idx] == 1:
            fraud_deg.append(deg)
        else:
            nonfraud_deg.append(deg)

    plt.figure()
    plt.hist(nonfraud_deg, bins=50, alpha=0.7, label="Non-Fraud")
    plt.hist(fraud_deg, bins=50, alpha=0.7, label="Fraud")
    plt.legend()
    plt.title("Transaction Degree: Fraud vs Non-Fraud")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/transaction_degree_fraud_vs_nonfraud.png")
    plt.close()

    print("Plots saved to:", output_dir)

if __name__ == "__main__":
    visualize()