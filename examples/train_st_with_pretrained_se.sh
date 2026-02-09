#!/bin/bash
#
# Esempio: Train State Transition (ST) model con SE pre-addestrato
#

# Percorso al checkpoint SE pre-addestrato
SE_CHECKPOINT="$HOME/PT-RAG/pretrained_SE/se600m_epoch16.ckpt"

# Percorso ai dati
DATA_CONFIG="$HOME/PT-RAG/examples/fewshot.toml"

# Output directory
OUTPUT_DIR="$HOME/PT-RAG/experiments/st_with_pretrained_se"

# Esegui il training
state tx train data.kwargs.toml_config_path="${DATA_CONFIG}" \
  data.kwargs.embed_key=X_hvg \
  data.kwargs.num_workers=12 \
  data.kwargs.batch_col=batch_var \
  data.kwargs.pert_col=target_gene \
  data.kwargs.cell_type_key=cell_type \
  data.kwargs.control_pert=TARGET1 \
  training.max_steps=40000 \
  training.val_freq=100 \
  training.ckpt_every_n_steps=100 \
  training.batch_size=8 \
  training.lr=1e-4 \
  model.kwargs.cell_set_len=64 \
  model.kwargs.hidden_dim=328 \
  model.kwargs.init_from="${SE_CHECKPOINT}" \
  model.kwargs.freeze_pert_backbone=true \
  model=state \
  wandb.tags="[pretrained_se,finetuning]" \
  output_dir="${OUTPUT_DIR}" \
  name="st_with_pretrained_se"

# Note:
# - model.kwargs.init_from: carica il checkpoint SE pre-addestrato
# - model.kwargs.freeze_pert_backbone=true: freezal il backbone (opzionale)
# - Puoi anche usare LoRA per un fine-tuning più efficiente
