import torch

def analyze_graph(data):
    print("\n" + "="*30)
    print("      GRAPH STATISTICS")
    print("="*30)
    
    # 1. Basic counts
    num_tx = data['transaction'].num_nodes
    num_cards = data['card'].num_nodes
    print(f"Total Transactions: {num_tx:,}")
    print(f"Total Unique Cards: {num_cards:,}")
    
    # 2. Fraud counts
    fraud_count = data['transaction'].y.sum().item()
    print(f"Fraudulent Transactions: {int(fraud_count):,} ({(fraud_count/num_tx)*100:.2f}%)")
    
    # 3. Card Connectivity (Degrees)
    edge_index = data['transaction', 'uses', 'card'].edge_index
    card_indices = edge_index[1]
    
    # Count how many times each card appears in the edge list
    unique_cards, counts = torch.unique(card_indices, return_counts=True)
    
    print("\n" + "="*30)
    print("      CARD CONNECTIVITY")
    print("="*30)
    print(f"Average Transactions per Card: {counts.float().mean():.2f}")
    print(f"Max Transactions on a Single Card: {counts.max().item()}")
    
    # 4. Shared Cards
    multi_use_cards = (counts > 1).sum().item()
    print(f"Cards used more than once: {multi_use_cards:,} ({(multi_use_cards/num_cards)*100:.2f}%)")
    print("="*30 + "\n")

if __name__ == "__main__":
    print("Loading graph data...")
    # Loading the full dataset safely
    data = torch.load('../data/processed/graph_data.pt', weights_only=False)
    analyze_graph(data)