#!/bin/bash

echo "=========================================="
echo "PII NER - Complete Workflow"
echo "=========================================="
echo ""

# Step 1: Generate data
echo "Step 1: Generating synthetic training data..."
python src/generate_data.py \
    --train_size 800 \
    --dev_size 150 \
    --output_dir data

echo ""
echo "Step 2: Training enhanced BERT + BiLSTM + CRF model..."
python src/train_enhanced.py \
    --model_name distilbert-base-uncased \
    --train data/train.jsonl \
    --dev data/dev.jsonl \
    --out_dir out \
    --batch_size 16 \
    --epochs 5 \
    --lr 3e-5 \
    --max_length 256 \
    --use_lstm \
    --use_crf \
    --grad_accum_steps 1

echo ""
echo "Step 3: Running predictions on dev set..."
python src/predict_enhanced.py \
    --model_dir out \
    --input data/dev.jsonl \
    --output out/dev_pred.json \
    --post_process

echo ""
echo "Step 4: Evaluating results..."
python src/eval_span_f1.py \
    --gold data/dev.jsonl \
    --pred out/dev_pred.json

echo ""
echo "Step 5: Measuring latency..."
python src/measure_latency.py \
    --model_dir out \
    --input data/dev.jsonl \
    --runs 50

echo ""
echo "=========================================="
echo "Workflow complete!"
echo "=========================================="