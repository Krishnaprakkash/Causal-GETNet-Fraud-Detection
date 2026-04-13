#!/usr/bin/env python3
"""
Fraud Detection Report Generator

Generates comprehensive HTML reports from fraud prediction results.
Can be used standalone or with the inference script output.

Input:  data/processed/fraud_predictions.csv
        data/processed/fraud_report.html (output)
Output: data/processed/fraud_report.html

Usage:
    python generate_report.py
    python generate_report.py --input data/processed/fraud_predictions.csv
    python generate_report.py --output custom_report.html
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_INPUT = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "fraud_predictions.csv")
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "fraud_report.html")
FRAUD_THRESHOLD = 0.5

ENTITY_COLS = [
    "card1", "card2", "card3", "card4", "card5", "card6",
    "ProductCD", "P_emaildomain", "addr1", "addr2", "dist1",
]


def compute_metrics(preds, threshold: float = 0.5):
    flagged = preds >= threshold
    n_flagged = flagged.sum()
    n_total = len(preds)
    fraud_rate = n_flagged / n_total * 100

    precision = n_flagged / n_total if n_total > 0 else 0
    recall = min(1.0, n_flagged / max(1, (preds >= 0.5).sum()))
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "total_transactions": n_total,
        "flagged_frauds": int(n_flagged),
        "fraud_percentage": float(fraud_rate),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }


def get_risk_level(prob):
    if prob >= 0.8:
        return "HIGH", "high-risk"
    elif prob >= 0.6:
        return "MEDIUM", "medium-risk"
    else:
        return "LOW", "low-risk"


def generate_report(preds, threshold: float = 0.5, title: str = "Fraud Detection Report"):
    print("Generating HTML report...")

    flagged_indices = np.where(preds >= threshold)[0]
    flagged = sorted(flagged_indices, key=lambda i: preds[i], reverse=True)[:1000]

    top_frauds = []
    for idx in flagged[:100]:
        risk_label, risk_class = get_risk_level(preds[idx])
        top_frauds.append({
            "index": int(idx),
            "fraud_probability": float(preds[idx]),
            "risk_level": risk_label,
            "risk_class": risk_class,
        })

    metrics = compute_metrics(preds, threshold)

    threshold_variations = []
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        m = compute_metrics(preds, t)
        threshold_variations.append({
            "threshold": t,
            "flagged": m["flagged_frauds"],
            "percentage": m["fraud_percentage"],
        })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f0f2f5;
            color: #333;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header p {{
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .header .timestamp {{
            margin-top: 20px;
            font-size: 0.9em;
            opacity: 0.8;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        .card:hover {{
            transform: translateY(-2px);
        }}
        .card.primary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        .card.danger {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }}
        .card.warning {{
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            color: white;
        }}
        .card.success {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
        }}
        .card h3 {{
            font-size: 0.9em;
            opacity: 0.8;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}
        .card .value {{
            font-size: 2.5em;
            font-weight: bold;
        }}
        .section {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
            color: #555;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }}
        tr:hover {{
            background-color: #f8f9fa;
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
        .threshold-table {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .threshold-item {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .threshold-item .t-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }}
        .threshold-item .t-count {{
            color: #555;
            margin-top: 5px;
        }}
        .threshold-item .t-pct {{
            color: #888;
            font-size: 0.9em;
        }}
        .progress-bar {{
            background: #e0e0e0;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
        }}
        .insights {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        .insight-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .insight-card h4 {{
            color: #667eea;
            margin-bottom: 10px;
        }}
        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 1.8em;
            }}
            .card .value {{
                font-size: 2em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Fraud Detection Report</h1>
            <p>Comprehensive analysis of fraudulent transaction patterns</p>
            <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>

        <div class="grid">
            <div class="card primary">
                <h3>Total Transactions</h3>
                <div class="value">{metrics['total_transactions']:,}</div>
            </div>
            <div class="card danger">
                <h3>Flagged Frauds</h3>
                <div class="value">{metrics['flagged_frauds']:,}</div>
            </div>
            <div class="card warning">
                <h3>Fraud Percentage</h3>
                <div class="value">{metrics['fraud_percentage']:.2f}%</div>
            </div>
            <div class="card success">
                <h3>Model Threshold</h3>
                <div class="value">{threshold}</div>
            </div>
        </div>

        <div class="section">
            <h2>Threshold Analysis</h2>
            <div class="threshold-table">
"""

    for t in threshold_variations:
        html += f"""
                <div class="threshold-item">
                    <div class="t-value">{t['threshold']}</div>
                    <div class="t-count">{t['flagged']:,} flagged</div>
                    <div class="t-pct">{t['percentage']:.2f}%</div>
                </div>
"""

    html += f"""
            </div>
        </div>

        <div class="section">
            <h2>Top Flagged Transactions</h2>
            <p style="margin-bottom: 20px; color: #666;">
                Showing top 50 transactions with highest fraud probability. 
                Total of {len(top_frauds)} high-risk transactions detected.
            </p>
            <table>
                <thead>
                    <tr>
                        <th>Transaction Index</th>
                        <th>Fraud Probability</th>
                        <th>Risk Level</th>
                        <th>Recommendation</th>
                    </tr>
                </thead>
                <tbody>
"""

    for fraud in top_frauds[:50]:
        if fraud['risk_level'] == "HIGH":
            recommendation = "Immediate Review Required"
        elif fraud['risk_level'] == "MEDIUM":
            recommendation = "Secondary Review"
        else:
            recommendation = "Log & Monitor"

        html += f"""
                    <tr>
                        <td>{fraud['index']:,}</td>
                        <td class="{fraud['risk_class']}">{fraud['fraud_probability']:.4f}</td>
                        <td class="{fraud['risk_class']}">{fraud['risk_level']}</td>
                        <td>{recommendation}</td>
                    </tr>
"""

    html += f"""
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Model Insights</h2>
            <div class="insights">
                <div class="insight-card">
                    <h4>Detection Rate</h4>
                    <p>
                        At threshold {threshold}, the model flags approximately 
                        {metrics['fraud_percentage']:.2f}% of all transactions as potentially fraudulent.
                        This translates to {metrics['flagged_frauds']:,} transactions requiring review.
                    </p>
                </div>
                <div class="insight-card">
                    <h4>Risk Distribution</h4>
                    <p>
                        High-risk transactions (≥80% probability): {sum(1 for f in top_frauds if f['risk_level'] == 'HIGH'):,}<br>
                        Medium-risk transactions (60-80%): {sum(1 for f in top_frauds if f['risk_level'] == 'MEDIUM'):,}<br>
                        Low-risk transactions (<60%): {sum(1 for f in top_frauds if f['risk_level'] == 'LOW'):,}
                    </p>
                </div>
                <div class="insight-card">
                    <h4>Recommended Actions</h4>
                    <p>
                        1. Review all HIGH risk transactions immediately<br>
                        2. Schedule secondary review for MEDIUM risk<br>
                        3. Monitor LOW risk for patterns<br>
                        4. Adjust threshold based on operational capacity
                    </p>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Methodology</h2>
            <p>
                This report is generated by the Causal-GETNet HeteroConv model, a heterogeneous graph neural network
                designed for fraud detection. The model analyzes transaction patterns and entity relationships to identify
                suspicious activities.
            </p>
            <p style="margin-top: 15px;">
                <strong>Model Architecture:</strong> HeteroConv (GATConv per entity type) → TransformerConv → Classifier<br>
                <strong>Entity Types:</strong> card1-6, ProductCD, P_emaildomain, addr1-2, dist1<br>
                <strong>Training:</strong> Temporal 80/20 split on historical transaction data
            </p>
        </div>
    </div>
</body>
</html>
"""

    return html


def main():
    parser = argparse.ArgumentParser(description="Generate Fraud Detection Report")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Input CSV path")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output HTML path")
    parser.add_argument("--threshold", type=float, default=FRAUD_THRESHOLD, help="Fraud threshold")
    parser.add_argument("--title", type=str, default="Fraud Detection Report", help="Report title")
    args = parser.parse_args()

    print("=" * 60)
    print("  Fraud Detection Report Generator")
    print("=" * 60)

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        print("Please run inference_heteroconv.py first to generate predictions.")
        sys.exit(1)

    df = pd.read_csv(args.input)
    preds = df["isFraud"].values

    print(f"Loaded {len(preds)} predictions from {args.input}")
    print(f"Threshold: {args.threshold}")

    html = generate_report(preds, args.threshold, args.title)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html)

    print(f"\nReport saved to: {args.output}")

    flagged = (preds >= args.threshold).sum()
    print(f"Flagged frauds: {flagged:,} ({flagged/len(preds)*100:.2f}%)")

    print("=" * 60)
    print("  Report Generation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()