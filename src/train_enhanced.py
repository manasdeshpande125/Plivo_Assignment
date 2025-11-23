"""
Enhanced training script with class weighting and focal loss option
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import numpy as np

from dataset import PIIDataset, collate_batch
from labels import LABELS, LABEL2ID, PII_LABELS


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)
    
    def forward(self, logits, labels):
        ce_loss = self.ce(logits.view(-1, logits.size(-1)), labels.view(-1))
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha[labels.view(-1)] * focal_loss
        
        return focal_loss.mean()


def compute_class_weights(train_path, label2id):
    """Compute class weights based on label distribution"""
    import json
    from collections import Counter
    
    label_counts = Counter()
    
    with open(train_path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            text = obj['text']
            
            # Create char-level tags
            char_tags = ['O'] * len(text)
            for e in obj.get('entities', []):
                s, end, lab = e['start'], e['end'], e['label']
                if s < 0 or end > len(text) or s >= end:
                    continue
                char_tags[s] = f"B-{lab}"
                for i in range(s + 1, end):
                    char_tags[i] = f"I-{lab}"
            
            for tag in char_tags:
                label_counts[tag] += 1
    
    # Convert to weights (inverse frequency)
    total = sum(label_counts.values())
    weights = {}
    
    for label in LABELS:
        count = label_counts.get(label, 1)
        # Inverse frequency with smoothing
        weights[label] = total / (len(LABELS) * count)
    
    # Boost PII entity weights
    for label in weights:
        if any(pii_label in label for pii_label in PII_LABELS):
            weights[label] *= 1.5  # 50% boost for PII
    
    # Normalize
    max_weight = max(weights.values())
    weights = {k: v / max_weight for k, v in weights.items()}
    
    weight_tensor = torch.tensor([weights[label] for label in LABELS], dtype=torch.float32)
    
    return weight_tensor


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--use_lstm", action="store_true", default=True)
    ap.add_argument("--use_crf", action="store_true", default=True)
    ap.add_argument("--use_class_weights", action="store_true", default=True)
    ap.add_argument("--use_focal_loss", action="store_true", default=False)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    return ap.parse_args()


# def evaluate(model, dev_dl, device):
#     """Quick evaluation on dev set"""
#     model.eval()
#     total_loss = 0
#     num_batches = 0
    
#     with torch.no_grad():
#         for batch in dev_dl:
#             input_ids = torch.tensor(batch["input_ids"], device=device)
#             attention_mask = torch.tensor(batch["attention_mask"], device=device)
#             labels = torch.tensor(batch["labels"], device=device)
            
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#             loss = outputs["loss"]
            
#             total_loss += loss.item()
#             num_batches += 1
    
#     avg_loss = total_loss / max(1, num_batches)
#     model.train()
#     return avg_loss

def evaluate(model, dev_dl, device, use_crf=False):
    """Quick evaluation on dev set"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dev_dl:
            input_ids = torch.tensor(batch["input_ids"], device=device)
            attention_mask = torch.tensor(batch["attention_mask"], device=device)
            labels = torch.tensor(batch["labels"], device=device)
            
            # FIX: Replace -100 padding with 0 for CRF
            if use_crf:
                labels = labels.masked_fill(labels == -100, 0)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / max(1, num_batches)
    model.train()
    return avg_loss


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Device: {args.device}")
    print(f"Model: {args.model_name}")
    print(f"Using BiLSTM: {args.use_lstm}")
    print(f"Using CRF: {args.use_crf}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load datasets
    train_ds = PIIDataset(args.train, tokenizer, LABELS, max_length=args.max_length, is_train=True)
    dev_ds = PIIDataset(args.dev, tokenizer, LABELS, max_length=args.max_length, is_train=False)
    
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )
    
    dev_dl = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )
    
    # Create model
    if args.use_lstm:
        from model_enhanced import create_model
        model = create_model(args.model_name, len(LABELS), use_lstm=True, use_crf=args.use_crf)
    else:
        from transformers import AutoModelForTokenClassification
        from labels import ID2LABEL
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_name,
            num_labels=len(LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
    
    model.to(args.device)
    model.train()
    
    # Optimizer with layer-wise learning rate decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and "bert" in n],
            "weight_decay": 0.01,
            "lr": args.lr * 0.1,  # Lower LR for BERT
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and "bert" in n],
            "weight_decay": 0.0,
            "lr": args.lr * 0.1,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if "bert" not in n],
            "weight_decay": 0.01,
            "lr": args.lr,  # Higher LR for task-specific layers
        },
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    
    total_steps = len(train_dl) * args.epochs // args.grad_accum_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )
    
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"Total steps: {total_steps}")
    
    best_dev_loss = float('inf')
    
    # for epoch in range(args.epochs):
    #     running_loss = 0.0
    #     optimizer.zero_grad()
        
    #     progress_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}")
        
    #     for step, batch in enumerate(progress_bar):
    #         input_ids = torch.tensor(batch["input_ids"], device=args.device)
    #         attention_mask = torch.tensor(batch["attention_mask"], device=args.device)
    #         labels = torch.tensor(batch["labels"], device=args.device)
            
    #         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    #         loss = outputs["loss"]
            
    #         # Gradient accumulation
    #         loss = loss / args.grad_accum_steps
    #         loss.backward()
            
    #         running_loss += loss.item() * args.grad_accum_steps
            
    #         if (step + 1) % args.grad_accum_steps == 0:
    #             # Gradient clipping
    #             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #             optimizer.step()
    #             scheduler.step()
    #             optimizer.zero_grad()
            
    #         progress_bar.set_postfix({"loss": f"{loss.item() * args.grad_accum_steps:.4f}"})

    for epoch in range(args.epochs):
        running_loss = 0.0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for step, batch in enumerate(progress_bar):
            input_ids = torch.tensor(batch["input_ids"], device=args.device)
            attention_mask = torch.tensor(batch["attention_mask"], device=args.device)
            labels = torch.tensor(batch["labels"], device=args.device)
            
            # FIX: Replace -100 padding with 0 (O label) for CRF
            if args.use_crf:
                labels = labels.masked_fill(labels == -100, 0)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            
            # Gradient accumulation
            loss = loss / args.grad_accum_steps
            loss.backward()
            
            running_loss += loss.item() * args.grad_accum_steps
            
            if (step + 1) % args.grad_accum_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.set_postfix({"loss": f"{loss.item() * args.grad_accum_steps:.4f}"})
            
        avg_loss = running_loss / max(1, len(train_dl))
        
        # Evaluate on dev
        # dev_loss = evaluate(model, dev_dl, args.device)
        # Evaluate on dev
        dev_loss = evaluate(model, dev_dl, args.device, use_crf=args.use_crf)
        
        print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Dev Loss: {dev_loss:.4f}")
        
        # Save best model
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            model_save_path = os.path.join(args.out_dir, "best_model")
            os.makedirs(model_save_path, exist_ok=True)
            
            # Save model state
            torch.save({
                'model_state_dict': model.state_dict(),
                'use_lstm': args.use_lstm,
                'use_crf': args.use_crf,
                'model_name': args.model_name,
                'num_labels': len(LABELS),
            }, os.path.join(model_save_path, "pytorch_model.bin"))
            
            tokenizer.save_pretrained(model_save_path)
            print(f"Saved best model to {model_save_path}")
    
    # Save final model
    final_save_path = args.out_dir
    torch.save({
        'model_state_dict': model.state_dict(),
        'use_lstm': args.use_lstm,
        'use_crf': args.use_crf,
        'model_name': args.model_name,
        'num_labels': len(LABELS),
    }, os.path.join(final_save_path, "pytorch_model.bin"))
    
    tokenizer.save_pretrained(final_save_path)
    print(f"\nTraining complete! Model saved to {args.out_dir}")


if __name__ == "__main__":
    main()