import torch

def detailed_connectivity_analysis(data):
    print("\n" + "="*55)
    print("   DETAILED GRAPH CONNECTIVITY & FRAUD ANALYSIS")
    print("="*55)

    # ---------------------------------------------------------
    # 1. Overall Dataset
    # ---------------------------------------------------------
    tx_labels = data['transaction'].y
    total_tx = len(tx_labels)
    total_fraud = tx_labels.sum().item()
    total_legit = total_tx - total_fraud
    total_cards = data['card'].num_nodes

    print(f"\n[1] OVERALL TRANSACTION SUMMARY")
    print(f"Total Transactions: {total_tx:,}")
    print(f"  - Not Fraud: {int(total_legit):,} ({(total_legit/total_tx)*100:.2f}%)")
    print(f"  - Fraud:     {int(total_fraud):,} ({(total_fraud/total_tx)*100:.2f}%)")
    print(f"Total Unique Cards: {total_cards:,}")

    # ---------------------------------------------------------
    # 2. Connection Count Buckets (Degrees)
    # ---------------------------------------------------------
    edge_index = data['transaction', 'uses', 'card'].edge_index
    tx_indices = edge_index[0]
    card_indices = edge_index[1]

    # Calculate transactions per card and fraud per card
    fraud_per_card = torch.zeros(total_cards)
    tx_per_card = torch.zeros(total_cards)
    
    fraud_per_card.scatter_add_(0, card_indices, tx_labels[tx_indices])
    tx_per_card.scatter_add_(0, card_indices, torch.ones_like(card_indices, dtype=torch.float))

    print(f"\n[2] HOW MANY TRANSACTIONS ARE CONNECTED TO EACH CARD?")
    print(f"  - Cards with exactly 1 transaction:  {(tx_per_card == 1).sum().item():,}")
    print(f"  - Cards with 2 to 10 transactions:   {((tx_per_card >= 2) & (tx_per_card <= 10)).sum().item():,}")
    print(f"  - Cards with 11 to 100 transactions: {((tx_per_card > 10) & (tx_per_card <= 100)).sum().item():,}")
    print(f"  - Cards with 101 to 1000 transactions: {((tx_per_card > 100) & (tx_per_card <= 1000)).sum().item():,}")
    print(f"  - Cards with > 1000 transactions:    {(tx_per_card > 1000).sum().item():,}")

    # ---------------------------------------------------------
    # 3. Fraud, Not Fraud, and Combined Buckets
    # ---------------------------------------------------------
    only_legit_cards = ((tx_per_card > 0) & (fraud_per_card == 0)).sum().item()
    only_fraud_cards = ((tx_per_card > 0) & (fraud_per_card == tx_per_card)).sum().item()
    mixed_cards = ((fraud_per_card > 0) & (fraud_per_card < tx_per_card)).sum().item()

    print(f"\n[3] CARD CATEGORIES (Clean, Fraudulent, or Combined)")
    print(f"  - CLEAN CARDS (0% Fraud):          {only_legit_cards:,}")
    print(f"  - COMPROMISED CARDS (100% Fraud):  {only_fraud_cards:,}")
    print(f"  - COMBINED CARDS (Mixed Fraud/Not):{mixed_cards:,}")
    print("="*55 + "\n")

if __name__ == "__main__":
    print("Loading graph data...")
    # Load the graph safely
    data = torch.load('../data/processed/graph_data.pt', weights_only=False)
    detailed_connectivity_analysis(data)