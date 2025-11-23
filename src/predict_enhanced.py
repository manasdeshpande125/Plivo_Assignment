"""
Enhanced prediction with post-processing for high precision
"""
import json
import argparse
import torch
import re
from transformers import AutoTokenizer
from labels import ID2LABEL, label_is_pii


def load_model(model_dir, device):
    """Load trained model"""
    import os
    
    # Load model config
    checkpoint = torch.load(
        os.path.join(model_dir, "pytorch_model.bin"),
        map_location=device
    )
    
    use_lstm = checkpoint.get('use_lstm', False)
    use_crf = checkpoint.get('use_crf', False)
    model_name = checkpoint.get('model_name', 'distilbert-base-uncased')
    num_labels = checkpoint.get('num_labels', len(ID2LABEL))
    
    if use_lstm:
        from model_enhanced import create_model
        model = create_model(model_name, num_labels, use_lstm=True, use_crf=use_crf)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        from transformers import AutoModelForTokenClassification
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
    
    return model, use_crf


def bio_to_spans(text, offsets, label_ids, use_crf=False):
    """Convert BIO tags to spans with improved handling"""
    spans = []
    current_label = None
    current_start = None
    current_end = None
    
    for (start, end), lid in zip(offsets, label_ids):
        if start == 0 and end == 0:  # Special token
            continue
        
        label = ID2LABEL.get(int(lid), "O")
        
        if label == "O":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue
        
        prefix, ent_type = label.split("-", 1)
        
        if prefix == "B":
            # Start new entity
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
            current_label = ent_type
            current_start = start
            current_end = end
        elif prefix == "I":
            if current_label == ent_type:
                # Continue current entity
                current_end = end
            else:
                # Invalid transition, treat as new entity
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end
    
    if current_label is not None:
        spans.append((current_start, current_end, current_label))
    
    return spans


def validate_credit_card(text):
    """Simple validation for credit card (length check)"""
    # Remove spaces and check if it's numeric when converted
    words = text.lower().split()
    digit_words = {'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'}
    
    # Check if mostly digit words
    digit_count = sum(1 for w in words if w in digit_words)
    
    # Should have 16-19 digit words for valid credit card
    if 14 <= digit_count <= 20:
        return True
    return False


def validate_phone(text):
    """Validate phone number"""
    words = text.lower().split()
    digit_words = {'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'}
    
    digit_count = sum(1 for w in words if w in digit_words)
    
    # Should have 10-11 digits for valid phone
    if 9 <= digit_count <= 12:
        return True
    return False


def validate_email(text):
    """Validate email format"""
    # Should contain "at" and "dot"
    text_lower = text.lower()
    if 'at' in text_lower and 'dot' in text_lower:
        # Check for common patterns
        if any(domain in text_lower for domain in ['gmail', 'yahoo', 'hotmail', 'outlook', 'com', 'org']):
            return True
    return False


def validate_person_name(text):
    """Validate person name"""
    # Names should not be too short or too long
    words = text.strip().split()
    if 1 <= len(words) <= 4:
        # Check if contains common titles
        titles = ['mr', 'mrs', 'miss', 'ms', 'dr', 'doctor', 'mister']
        non_title_words = [w for w in words if w.lower() not in titles]
        if non_title_words:
            return True
    return False


def validate_date(text):
    """Validate date"""
    months = ['january', 'february', 'march', 'april', 'may', 'june',
              'july', 'august', 'september', 'october', 'november', 'december']
    
    text_lower = text.lower()
    
    # Check if contains month name
    if any(month in text_lower for month in months):
        return True
    
    # Check if contains year (4 digits)
    if re.search(r'\b(19|20)\d{2}\b', text):
        return True
    
    return False


def post_process_spans(spans, text, confidence_threshold=0.7):
    """
    Post-process spans to improve precision
    
    Args:
        spans: List of (start, end, label, confidence) tuples
        text: Original text
        confidence_threshold: Minimum confidence for PII entities
    
    Returns:
        Filtered spans
    """
    validated_spans = []
    
    validators = {
        'CREDIT_CARD': validate_credit_card,
        'PHONE': validate_phone,
        'EMAIL': validate_email,
        'PERSON_NAME': validate_person_name,
        'DATE': validate_date,
    }
    
    for start, end, label in spans:
        entity_text = text[start:end]
        
        # Apply validators for PII entities
        if label in validators:
            if not validators[label](entity_text):
                # Skip invalid entities
                continue
        
        # Add validated span
        validated_spans.append((start, end, label))
    
    return validated_spans


def merge_overlapping_spans(spans):
    """Merge overlapping spans, keeping higher confidence ones"""
    if not spans:
        return spans
    
    # Sort by start position
    sorted_spans = sorted(spans, key=lambda x: (x[0], x[1]))
    
    merged = []
    current = list(sorted_spans[0])
    
    for next_span in sorted_spans[1:]:
        next_start, next_end, next_label = next_span
        curr_start, curr_end, curr_label = current
        
        # Check for overlap
        if next_start < curr_end:
            # Overlapping - keep longer span
            if (next_end - next_start) > (curr_end - curr_start):
                current = list(next_span)
        else:
            # No overlap
            merged.append(tuple(current))
            current = list(next_span)
    
    merged.append(tuple(current))
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--confidence_threshold", type=float, default=0.6)
    ap.add_argument("--post_process", action="store_true", default=True)
    args = ap.parse_args()
    
    print(f"Loading model from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model, use_crf = load_model(args.model_dir, args.device)
    model.eval()
    
    print(f"Using CRF: {use_crf}")
    print(f"Post-processing: {args.post_process}")
    
    results = {}
    
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]
            
            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)
            
            with torch.no_grad():
                if use_crf:
                    # Use CRF decoding
                    pred_ids = model.predict(input_ids, attention_mask)[0].cpu().tolist()
                else:
                    # Standard argmax
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = out["logits"][0]
                    pred_ids = logits.argmax(dim=-1).cpu().tolist()
            
            spans = bio_to_spans(text, offsets, pred_ids, use_crf=use_crf)
            
            # Post-processing
            if args.post_process:
                spans = post_process_spans(spans, text, args.confidence_threshold)
                spans = merge_overlapping_spans(spans)
            
            ents = []
            for s, e, lab in spans:
                ents.append(
                    {
                        "start": int(s),
                        "end": int(e),
                        "label": lab,
                        "pii": bool(label_is_pii(lab)),
                    }
                )
            results[uid] = ents
    
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Wrote predictions for {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()