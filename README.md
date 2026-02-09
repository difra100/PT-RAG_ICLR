# 🧬 PT-RAG: Differentiable Retrieval-Augmented Generation for Single-Cell Perturbation Prediction

## 📋 Overview

We propose **PT-RAG** (Perturbation-aware Two-Stage Retrieval-Augmented Generation), a novel differentiable RAG pipeline that enhances single-cell perturbation response generation. PT-RAG leverages similar perturbation examples from a training database through a fully differentiable retrieval mechanism with sparsity regularization.

!image[figures/ptrag_scheme_final.png]  

### Key Contributions

- **🎯 Differentiable Retrieval**: End-to-end learnable retrieval with Gumbel-softmax sparsity
- **🧠 Cell-Type Aware Selection**: Adaptive retrieval strategy based on cell type characteristics  
- **⚖️ Sparsity Regularization**: Learned sparse attention over retrieved examples
- **🔄 Retrieve-then-Predict**: Two-stage pipeline optimizing retrieval for downstream prediction

### Model Architecture

- **SE (State Embedding)**: Generates embeddings from gene expression profiles
- **ST (State Transition)**: Predicts cellular responses to perturbations  
- **PT-RAG** ⭐: Our proposed differentiable RAG-enhanced state transition model

## ⚡ Installation

### Prerequisites

- Python 3.11
- CUDA-compatible GPU (recommended for training)

### Setup Instructions

#### 1. Create Conda Environment (Recommended)

```bash
# Create a new conda environment
conda create -n ptrag python=3.11 -y
conda activate ptrag
```

#### 2. Install Base Package

```bash
cd /path/to/ptrag
pip install -e .
```

#### 3. Install RAG Dependencies

To enable the RAG feature, you must install additional dependencies:

```bash
pip install -r requirements.txt
```

This will install:
- `faiss-cpu` or `faiss-gpu` for efficient similarity search
- Additional dependencies for retrieval-augmented generation



### Download Dataset

```bash
mkdir datasets  

# Download Replogle-Nadig dataset
bash datasets/get_repogle_nadig.sh
```

### Download GenePT
```bash
mkdir genept_emb  

# Download Replogle-Nadig dataset
bash genept_emb/get_genept_emb.sh
```


### Download Cell Pre-Trained Encoder

Link at https://huggingface.co/arcinstitute/SE-600M  
```bash  
mkdir pretrained_SE  
```
Put the model weights in pretrained_SE/.  


## 🎯 PT-RAG: Differentiable Retrieval Pipeline

PT-RAG introduces a fully differentiable retrieval mechanism that learns to select and weight relevant perturbation examples for improved predictions.

### How PT-RAG Works

1. **🔍 Differentiable Retrieval**: Learn retrieval weights end-to-end using Gumbel-softmax
2. **🎯 Sparsity Regularization**: Encourage focused attention on most relevant examples  
3. **🧠 Context Integration**: Cross-attention mechanism fuses retrieved examples with query
4. **⚖️ Cell-Type Adaptation**: Retrieve-then-predict strategy adapts to cell type characteristics
5. **📊 Joint Optimization**: Retrieval and prediction trained jointly for optimal performance

### Key Parameters

- `training.differentiable_rag=true`: Enable differentiable retrieval mechanism
- `training.retrieve_than_predict=true`: Use two-stage retrieve-then-predict pipeline
- `training.gumbel_sparsity_loss=true`: Apply sparsity regularization to attention weights
- `training.gumbel_sparsity_weight=0.1`: Weight for sparsity regularization (0.01-0.1)
- `training.topk_rag=32`: Number of candidates for differentiable selection

### Training PT-RAG

To train PT-RAG with differentiable retrieval and sparsity regularization:

```bash
python -m state.__main__ tx train \
    data.kwargs.toml_config_path=datasets/repogle_nadig_jurkat.toml \
    training.rag=true \
    training.differentiable_rag=true \
    training.retrieve_than_predict=true \
    training.gumbel_sparsity_loss=true \
    training.gumbel_sparsity_weight=0.1 \
    training.topk_rag=32 \
    training.use_genept=true \
    model=state \
    output_dir=experiments/ptrag_model \
    name=jurkat_ptrag_sparsity0.1
```

### Inference with PT-RAG

The differentiable RAG index and learned weights are automatically saved during training and loaded during inference:

```bash
python -m state.__main__ tx predict \
    --output-dir experiments/ptrag_model \
    --checkpoint last.ckpt \
    --eval-genept-pert
```

## 🚀 Training Scripts

### Multi-Cell-Type Training

Train PT-RAG across multiple cell types using our provided script:

```bash
# Run experiments on RPE1, Jurkat, and K562 cell lines
bash run_celltype_experiments.sh
```

This script trains:
- **PT-RAG models**: With differentiable retrieval + sparsity 0.1
- **Baseline models**: Without RAG for comparison

### Cell-Type Specific Training

Train on individual cell types:

```bash
# RPE1 with PT-RAG
python -m state.__main__ tx train \
    data.kwargs.toml_config_path=datasets/repogle_nadig_rpe1.toml \
    training.rag=true \
    training.differentiable_rag=true \
    training.retrieve_than_predict=true \
    training.gumbel_sparsity_loss=true \
    training.gumbel_sparsity_weight=0.1 \
    training.topk_rag=32 \
    name=rpe1_ptrag_sparsity0.1

# Baseline without RAG
python -m state.__main__ tx train \
    data.kwargs.toml_config_path=datasets/repogle_nadig_rpe1.toml \
    name=rpe1_baseline
```

## 🔬 Predictions and Evaluation

### Batch Predictions

Run predictions on multiple trained models:

```bash
# Generate predictions for all trained models
bash run_replogle_predictions.sh
```

### Individual Predictions

Run predictions on specific models:

```bash
python -m state.__main__ tx predict \
    --output-dir experiments/replogle/jurkat_ptrag_sparsity0.1 \
    --checkpoint last.ckpt \
    --eval-genept-pert \
    --count-flops
```

**Prediction Options:**
- `--eval-genept-pert`: Evaluate on GenePT perturbations  
- `--count-flops`: Count floating point operations for efficiency analysis

## 📚 Quick Start Examples


### Train Baseline Model (No RAG)

```bash
python -m state.__main__ tx train \
    data.kwargs.toml_config_path=datasets/repogle_nadig_jurkat.toml \
    model=state \
    output_dir=experiments/baseline \
    name=jurkat_baseline
```

### Train PT-RAG Model

```bash
python -m state.__main__ tx train \
    data.kwargs.toml_config_path=datasets/repogle_nadig_jurkat.toml \
    training.rag=true \
    training.differentiable_rag=true \
    training.retrieve_than_predict=true \
    training.gumbel_sparsity_loss=true \
    training.gumbel_sparsity_weight=0.1 \
    training.topk_rag=32 \
    training.use_genept=true \
    model=state \
    output_dir=experiments/ptrag \
    name=jurkat_ptrag
```

### Run Predictions and Analysis

```bash
# Generate predictions
python -m state.__main__ tx predict \
    --output-dir experiments/ptrag \
    --checkpoint last.ckpt \
    --eval-genept-pert



## 🙏 Acknowledgments

This work builds upon the original State model and introduces PT-RAG, a novel differentiable retrieval-augmented generation approach for improved single-cell perturbation modeling.