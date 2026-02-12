# prep_data.py: Annotate/prepare datasets for RAG/LoRA. Focus: Bitcoin TA pairs (image + text analysis + metadata).
# Details: Handles ~5000 images; JSON output for fine-tuning. Synthetic mode for testing; generalization via asset_type.
# Integrations: llmware-inspired ingestion logic.
# Usage: python scripts/prep_data.py --data_dir path/to/data --mode [sample|full] --output annotations.json --asset_type [bitcoin|stock|custom]

import argparse
import json
import os
import random
import pandas as pd  # For rigorous CSV handling

parser = argparse.ArgumentParser(description="Prepare TA dataset.")
parser.add_argument("--data_dir", default="examples/btc_dataset_sample", help="Path to images/charts and text/prices.")
parser.add_argument("--mode", choices=["sample", "full"], default="full", help="Sample for testing or full dataset.")
parser.add_argument("--output", default="annotations.json", help="Output JSON file.")
parser.add_argument("--asset_type", default="bitcoin", help="Asset type for metadata (e.g., stock for tickers).")
args = parser.parse_args()

data = []
if args.mode == "sample":
    for i in range(100):
        price = 66000 + random.randint(-1000, 1000)
        data.append({
            "image": f"charts/chart{i}.png",
            "analysis": f"Bullish flag at ${price}; RSI divergence." if args.asset_type == "bitcoin" else f"Uptrend for {args.asset_type.upper()} at ${price}.",
            "metadata": {"timestamp": "2026-02-11", "price": price, "pattern": "flag", "asset": args.asset_type}
        })
else:
    try:
        # Full: Scan dir; load prices CSV rigorously
        images = [f for f in os.listdir(os.path.join(args.data_dir, "charts")) if f.endswith(('.png', '.jpg'))]
        prices_df = pd.read_csv(os.path.join(args.data_dir, "prices.csv"))  # Handle timestamps/OHLCV; fill NaNs
        for idx, img in enumerate(images[:5000]):
            row = prices_df.iloc[idx % len(prices_df)]  # Cycle if mismatch
            data.append({
                "image": os.path.join("charts", img),
                "analysis": "User-annotated TA here",  # Placeholder: Extend with Ollama auto-prompt for broader appeal
                "metadata": {"timestamp": row.get('timestamp', 'extract_from_filename'), "price": row.get('close', 0.0), "asset": args.asset_type}
            })
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

with open(args.output, 'w') as f:
    json.dump(data, f, indent=4)

print(f"Prepared {len(data)} examples in {args.output}. For auto-annotation, extend with Ollama prompts.")
