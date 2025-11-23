import json
import time
import argparse
import statistics
import os
import torch
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"Loading model from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    
    # Check if enhanced model exists
    checkpoint_path = os.path.join(args.model_dir, "pytorch_model.bin")
    if os.path.exists(checkpoint_path):
        # Load enhanced model
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        use_lstm = checkpoint.get('use_lstm', False)
        use_crf = checkpoint.get('use_crf', False)
        model_name = checkpoint.get('model_name', 'distilbert-base-uncased')
        num_labels = checkpoint.get('num_labels', 15)
        
        if use_lstm:
            # Import enhanced model
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
            from model_enhanced import create_model
            model = create_model(model_name, num_labels, use_lstm=True, use_crf=use_crf)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    else:
        # Fallback to standard model
        model = AutoModelForTokenClassification.from_pretrained(args.model_dir)

    
    model.to(args.device)
    model.eval()
    

    texts = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])

    if not texts:
        print("No texts found in input file.")
        return

    times_ms = []

    # warmup
    for _ in range(5):
        t = texts[0]
        enc = tokenizer(
            t,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            _ = model(input_ids=enc["input_ids"].to(args.device), attention_mask=enc["attention_mask"].to(args.device))

    for i in range(args.runs):
        t = texts[i % len(texts)]
        enc = tokenizer(
            t,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids=enc["input_ids"].to(args.device), attention_mask=enc["attention_mask"].to(args.device))
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    p50 = statistics.median(times_ms)
    times_sorted = sorted(times_ms)
    p95 = times_sorted[int(0.95 * len(times_sorted)) - 1]

    print(f"Latency over {args.runs} runs (batch_size=1):")
    print(f"  p50: {p50:.2f} ms")
    print(f"  p95: {p95:.2f} ms")


if __name__ == "__main__":
    main()

