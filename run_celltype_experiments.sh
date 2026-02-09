#!/bin/bash
set -e  # Exit on error

echo "=========================================="
echo "Running Replogle Cell Type Experiments"
echo "=========================================="

# Base directory for experiments
EXP_DIR="$HOME/PT-RAG/experiments/replogle"
PRETRAINED_SE="$HOME/PT-RAG/pretrained_SE/se600m_epoch16.ckpt"

# Cell types and their corresponding TOML files
declare -A CELL_TYPE_CONFIGS=(
    ["hepg2"]="$HOME/PT-RAG/datasets/repogle_nadig.toml"
    ["rpe1"]="$HOME/PT-RAG/datasets/repogle_nadig_rpe1.toml"
    ["jurkat"]="$HOME/PT-RAG/datasets/repogle_nadig_jurkat.toml"
    ["k562"]="$HOME/PT-RAG/datasets/repogle_nadig_k562.toml"
)

# Common arguments for all experiments
COMMON_ARGS=(
    "tx"
    "train"
    "data.kwargs.embed_key=X_hvg"
    "data.kwargs.num_workers=4"
    "data.kwargs.output_space=gene"
    "data.kwargs.batch_col=gem_group"
    "data.kwargs.pert_col=gene"
    "data.kwargs.cell_type_key=cell_line"
    "data.kwargs.control_pert=non-targeting"
    "training.max_steps=50001"
    "training.val_freq=2000"
    "training.ckpt_every_n_steps=5000"
    "training.batch_size=64"
    "training.lr=1e-3"
    "model.kwargs.cell_set_len=64"
    "model.kwargs.hidden_dim=128"
    "model.kwargs.batch_encoder=true"
    "model.kwargs.init_from=$PRETRAINED_SE"
    "model.kwargs.freeze_pert_backbone=true"
    "model=state"
    "wandb.tags=[replogle_run,pretrained_se,celltype_comparison]"
    "output_dir=$EXP_DIR"
    "use_wandb=true"
)

# Iterate over each cell type
for CELL_TYPE in rpe1 jurkat k562; do
    TOML_CONFIG="${CELL_TYPE_CONFIGS[$CELL_TYPE]}"
    
    echo ""
    echo "=========================================="
    echo "Processing cell type: $CELL_TYPE"
    echo "TOML config: $TOML_CONFIG"
    echo "=========================================="
    
    # Experiment 1: RAG with differentiable retrieval (sparsity 0.1)
    echo ""
    echo "Running ${CELL_TYPE} with RAG + differentiable + sparsity 0.1"
    python -m state.__main__ "${COMMON_ARGS[@]}" \
        "data.kwargs.toml_config_path=$TOML_CONFIG" \
        "training.rag=true" \
        "training.differentiable_rag=true" \
        "training.retrieve_than_predict=true" \
        "training.use_genept=true" \
        "training.gumbel_sparsity_loss=true" \
        "training.gumbel_sparsity_weight=0.1" \
        "training.topk_rag=32" \
        "name=${CELL_TYPE}_rag_diff_sparsity0.1" \
        "wandb.tags=[replogle_run,rag,differentiable,sparsity_0.1,${CELL_TYPE},pretrained_se]"
    
    # Experiment 2: No RAG (baseline)
    echo ""
    echo "Running ${CELL_TYPE} without RAG (baseline)"
    python -m state.__main__ "${COMMON_ARGS[@]}" \
        "data.kwargs.toml_config_path=$TOML_CONFIG" \
        "name=${CELL_TYPE}_no_rag_baseline" \
        "wandb.tags=[replogle_run,no_rag,baseline,${CELL_TYPE},pretrained_se]"
    
    echo ""
    echo "Completed experiments for $CELL_TYPE"
    echo "=========================================="
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Summary:"
echo "- 3 cell types: rpe1, jurkat, k562"
echo "- 2 configurations per cell type:"
echo "  1. RAG + differentiable + sparsity 0.1"
echo "  2. No RAG (baseline)"
echo "- Total: 6 experiments"
echo "- Steps per experiment: 30,000"
echo "=========================================="
