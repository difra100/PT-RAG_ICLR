echo "=========================================="
echo "Running Predictions on Replogle Models"
echo "=========================================="

# Base directory for experiments
EXP_DIR="$HOME/PT-RAG/experiments/replogle"

# Array of model directories
# MODELS=(
#     "hepg2_holdout_rag_genept_retrieve_than_predict"
#     "hepg2_holdout_rag_top2genept_retrieve_than_predict"
#     "hepg2_holdout_rag_top5genept_retrieve_than_predict"
#     "hepg2_holdout_rag_top10genept_retrieve_than_predict"

MODELS=(
    # "hepg2_holdout"
    # "hepg2_rag32_sparsity_0.1"
    "k562_no_rag_genept_baseline"
    "rpe1_rag_genept_retrieve_k32"
    # "hepg2_holdout_rag_genept_retrieve_than_predict"
    # "hepg2_rag_genept_retrieve_k32"
    # "hepg2_no_rag_genept_baseline"
    
    # rpe1_rag_diff_sparsity0.1
    # k562_no_rag_baseline
    # jurkat_rag_genept_retrieve_k32
    # jurkat_no_rag_genept_baseline
    # rpe1_no_rag_genept_baseline
    # k562_rag_diff_sparsity0.1
    # "hepg2_holdout_rag_top2genept_retrieve_than_predict"
    # "hepg2_holdout_rag_top5genept_retrieve_than_predict"
    # "hepg2_holdout_rag_top10genept_retrieve_than_predict"
    # "hepg2_rag32_sparsity_0.0"
    # "hepg2_rag32_sparsity_0.01"


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
