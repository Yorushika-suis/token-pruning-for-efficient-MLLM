#!/usr/bin/env bash
# First-protocol MetaCompress train + testdev eval.
# Unpublished paper choices are the documented first-version hypotheses:
# m=57, d_c=256, max pooling, sinusoidal PE, short-answer teacher forcing,
# generated history, image only on turn 1, seed 0.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-/root/Test/envs/metacompress/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Python interpreter not found or not executable: $PYTHON" >&2
  echo "Expected the migrated env at /root/Test/envs/metacompress. Set PYTHON to override." >&2
  exit 1
fi
MODEL="${MODEL:-$ROOT/weights/hub/models--liuhaotian--llava-v1.5-7b/snapshots/haotian-from-hf}"
TRAIN_DIALOGS="${TRAIN_DIALOGS:-$ROOT/weights/datasets/gqa/processed/mt_gqa/mt_gqa_train_balanced_20000.json}"
TRAIN_IMAGES="${TRAIN_IMAGES:-$ROOT/weights/datasets/gqa/processed/mt_gqa/images}"
DIALOGS="${DIALOGS:-$ROOT/weights/datasets/gqa/processed/mt_gqa_testdev/mt_gqa_testdev_balanced_4061.json}"
IMAGE_ROOT="${IMAGE_ROOT:-$ROOT/ThirdParty/LLaVA/playground/data/eval/gqa/data/images}"
TEST_QIDS="${TEST_QIDS:-$ROOT/weights/datasets/gqa/processed/mt_gqa_testdev/question_ids.txt}"
TEST_IIDS="${TEST_IIDS:-$ROOT/weights/datasets/gqa/processed/mt_gqa_testdev/image_ids.txt}"
RUN_DIR="${RUN_DIR:-$ROOT/outputs/metacompress/mt_gqa}"
SEED="${SEED:-0}"
GPU="${GPU:-0}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$ROOT/weights}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

mkdir -p "$RUN_DIR"
TRAIN_OUT="$RUN_DIR/train_seed_${SEED}"

"$PYTHON" "$ROOT/scripts/metacompress/train_llava_metacompress.py" \
  --model-path "$MODEL" \
  --dialogs "$TRAIN_DIALOGS" \
  --image-root "$TRAIN_IMAGES" \
  --output "$TRAIN_OUT" \
  --testdev-question-ids "$TEST_QIDS" \
  --testdev-image-ids "$TEST_IIDS" \
  --target-tokens 57 \
  --compression-dim 256 \
  --pool-mode max \
  --position-embedding sinusoidal \
  --epochs 2 \
  --batch-size 36 \
  --learning-rate 1e-3 \
  --max-grad-norm 1e-2 \
  --seed "$SEED"

"$PYTHON" "$ROOT/scripts/metacompress/run_mt_gqa.py" \
  --model-path "$MODEL" \
  --dialogs "$DIALOGS" \
  --image-root "$IMAGE_ROOT" \
  --method metacompress \
  --target-tokens 57 \
  --compressor "$TRAIN_OUT/checkpoint-final" \
  --output "$RUN_DIR/metacompress_seed_${SEED}_predictions.jsonl"

"$PYTHON" "$ROOT/scripts/metacompress/score_mt_gqa.py" \
  --dialogs "$DIALOGS" \
  --predictions "$RUN_DIR/metacompress_seed_${SEED}_predictions.jsonl" \
  --output "$RUN_DIR/metacompress_seed_${SEED}_metrics.json"
