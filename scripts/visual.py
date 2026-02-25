import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def visualize_three_cases(data):
    print("Analyzing graph to find the best examples...")
    edge_index = data['transaction', 'uses', 'card'].edge_index
    tx_labels = data['transaction'].y
    tx_features = data['transaction'].x # We need this to get the amounts back
    
    tx_indices = edge_index[0]
    card_indices = edge_index[1]
    
    num_cards = data['card'].num_nodes
    fraud_per_card = torch.zeros(num_cards)
    tx_per_card = torch.zeros(num_cards)
    
    fraud_per_card.scatter_add_(0, card_indices, tx_labels[tx_indices])
    tx_per_card.scatter_add_(0, card_indices, torch.ones_like(card_indices, dtype=torch.float))
    
    # 1. 100% Fraud Card
    pure_fraud_mask = (fraud_per_card == tx_per_card) & (tx_per_card >= 3)
    pure_fraud_card = torch.where(pure_fraud_mask)[0][0].item()

    # 2. Mixed Card 
    mixed_mask = (fraud_per_card >= 2) & (fraud_per_card < tx_per_card) & (tx_per_card >= 6)
    mixed_card = torch.where(mixed_mask)[0][0].item()

    # 3. 100% Clean Card 
    clean_mask = (fraud_per_card == 0) & (tx_per_card == 8)
    clean_card = torch.where(clean_mask)[0][0].item()

    cards_to_plot = {
        f"100% Fraud\n(Compromised Card)": pure_fraud_card,
        f"Mixed Fraud & Legit\n(Account Takeover)": mixed_card,
        f"100% Innocent\n(Normal User)": clean_card
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.canvas.manager.set_window_title('Fraud Graph Topologies & Amounts')
    
    for ax, (title, card_id) in zip(axes, cards_to_plot.items()):
        G = nx.Graph()
        
        # Add the Central Card Node
        card_node = f"Card_{card_id}"
        G.add_node(card_node, color='#90EE90', size=2500, label=f"Card\n{card_id}")
        
        connected_txs = tx_indices[card_indices == card_id]
        
        for tx in connected_txs:
            tx_id = tx.item()
            is_fraud = tx_labels[tx_id].item()
            
            # REVERSE THE LOG MATH to get real dollars
            log_amt = tx_features[tx_id, 0].item()
            real_amt = np.expm1(log_amt)
            
            color = '#FF9999' if is_fraud == 1 else '#99CCFF'
            
            # Create a label with both ID and Amount
            tx_node = f"Tx_{tx_id}"
            label_text = f"Tx {tx_id}\n${real_amt:.2f}"
            
            G.add_node(tx_node, color=color, size=1800, label=label_text)
            G.add_edge(card_node, tx_node)
            
        colors = [node[1]['color'] for node in G.nodes(data=True)]
        sizes = [node[1]['size'] for node in G.nodes(data=True)]
        labels = nx.get_node_attributes(G, 'label')
        
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, ax=ax, node_color=colors, node_size=sizes, edge_color='gray')
        
        # Draw the custom labels
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
    plt.tight_layout()
    print("Plotting the 3 topologies with amounts...")
    plt.show()

if __name__ == "__main__":
    print("Loading graph data...")
    data = torch.load('../data/processed/graph_data.pt', weights_only=False)
    visualize_three_cases(data)