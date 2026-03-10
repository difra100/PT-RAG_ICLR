echo "=========================================="
echo "Running Predictions on Replogle Models"
echo "=========================================="

# Base directory for experiments
EXP_DIR="$HOME/PT-RAG/experiments/replogle"


MODELS=(
    "hepg2_rag32_sparsity_0.1"
    
)

# Checkpoint to use (default: last.ckpt)
CHECKPOINT="${1:-last.ckpt}"

echo ""
echo "Using checkpoint: $CHECKPOINT"
echo ""

# Run predictions for each model
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    NUM=$((i + 1))
    TOTAL="${#MODELS[@]}"
    
    echo "=========================================="
    echo "Prediction $NUM/$TOTAL: $MODEL"
    echo "=========================================="
    
    OUTPUT_DIR="$EXP_DIR/$MODEL"
    
    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "⚠️  Warning: Model directory not found: $OUTPUT_DIR"
        echo "   Skipping..."
        continue
    fi
    
    if [ ! -f "$OUTPUT_DIR/checkpoints/$CHECKPOINT" ]; then
        echo "⚠️  Warning: Checkpoint not found: $OUTPUT_DIR/checkpoints/$CHECKPOINT"
        echo "   Skipping..."
        continue
    fi
    
    echo "Running prediction on $MODEL..."
    PYTHONPATH=/home/PT-RAG/src conda run --no-capture-output -n state python -m state.__main__ tx predict \
        --output-dir "$OUTPUT_DIR" \
        --checkpoint "$CHECKPOINT" \
        --eval-genept-pert \
        --count-flops \
    
    echo "✅ Prediction completed for $MODEL"
    echo ""
done

# ========================================
# Summary
# ========================================
echo "=========================================="
echo "All predictions completed!"
echo "=========================================="
echo ""
echo "Results saved in:"
for MODEL in "${MODELS[@]}"; do
    EVAL_DIR="$EXP_DIR/$MODEL/eval_$CHECKPOINT"
    if [ -d "$EVAL_DIR" ]; then
        echo "  ✅ $EVAL_DIR"
    else
        echo "  ❌ $EVAL_DIR (not found)"
    fi
done
echo ""
echo "To compare results, check:"
echo "  - *_results.csv (per-perturbation metrics)"
echo "  - *_agg_results.csv (aggregate metrics)"
echo ""
