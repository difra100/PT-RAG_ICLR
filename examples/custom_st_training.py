"""
Esempio di come modificare e customizzare il training del modello ST con SE pre-addestrato.

Questo script mostra come:
1. Caricare un checkpoint SE pre-addestrato
2. Freezare parti del modello per fare fine-tuning
3. Aggiungere custom callbacks
4. Modificare il training loop

Autore: Esempio per sviluppo
"""

import os
import torch
import lightning.pytorch as pl
from pathlib import Path
from omegaconf import OmegaConf

# Import delle classi necessarie dal progetto state
from state.tx.models.state_transition import StateTransitionPerturbationModel
from cell_load.data_modules import PerturbationDataModule


def load_pretrained_se_and_train_st(
    se_checkpoint_path: str,
    data_config_path: str,
    output_dir: str,
    freeze_encoder: bool = True,
    use_lora: bool = False,
):
    """
    Carica un modello SE pre-addestrato e allena solo il modello ST.
    
    Args:
        se_checkpoint_path: Path al checkpoint SE pre-addestrato (.ckpt)
        data_config_path: Path alla configurazione dati (TOML)
        output_dir: Directory dove salvare i risultati
        freeze_encoder: Se True, freezal'encoder del modello
        use_lora: Se True, usa LoRA per fine-tuning efficiente
    """
    
    # 1. Carica il checkpoint SE pre-addestrato
    print(f"Caricamento checkpoint SE da: {se_checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(se_checkpoint_path, map_location=device, weights_only=False)
    
    # Estrai hyperparameters dal checkpoint SE (se disponibili)
    se_hparams = checkpoint.get("hyper_parameters", {})
    print(f"Hyperparameters SE: {se_hparams}")
    
    # 2. Crea il modello ST con configurazione appropriata
    model_config = {
        "input_dim": 2000,  # Dimensione input (numero di geni HVG)
        "hidden_dim": 512,  # Dimensione hidden
        "output_dim": 128,  # Dimensione output/latent
        "pert_dim": 512,    # Dimensione embedding perturbazione
        "batch_dim": None,  # Opzionale: dimensione batch embedding
        "cell_set_len": 64, # Lunghezza sequenza cellule
        "predict_residual": True,
        "distributional_loss": "energy",
        "transformer_backbone_key": "GPT2",
        "transformer_backbone_kwargs": {
            "n_positions": 64,
            "n_embd": 512,
            "n_layer": 6,
            "n_head": 8,
        },
        "output_space": "gene",  # o "all" per tutto il trascrittoma
        "gene_dim": 2000,
        "dropout": 0.1,
        "n_encoder_layers": 2,
        "n_decoder_layers": 2,
        "embed_key": "X_hvg",
        "control_pert": "non-targeting",
    }
    
    # Aggiungi configurazione LoRA se richiesta
    if use_lora:
        model_config["lora"] = {
            "enable": True,
            "r": 8,           # Rank della decomposizione LoRA
            "alpha": 16,      # Scaling factor
            "dropout": 0.1,   # Dropout per LoRA
        }
    
    # Crea il modello ST
    print("Creazione modello ST...")
    model = StateTransitionPerturbationModel(**model_config)
    
    # 3. Carica i pesi dal checkpoint SE
    print("Caricamento pesi dal checkpoint SE...")
    model_state = model.state_dict()
    checkpoint_state = checkpoint["state_dict"]
    
    # Filtra i parametri che hanno dimensioni compatibili
    filtered_state = {}
    for name, param in checkpoint_state.items():
        if name in model_state:
            if param.shape == model_state[name].shape:
                filtered_state[name] = param
                print(f"✓ Caricato: {name} {param.shape}")
            else:
                print(f"✗ Skipped (shape mismatch): {name} checkpoint={param.shape}, model={model_state[name].shape}")
        else:
            print(f"✗ Skipped (not in model): {name}")
    
    # Carica i pesi filtrati (strict=False permette caricamento parziale)
    model.load_state_dict(filtered_state, strict=False)
    print(f"Caricati {len(filtered_state)}/{len(checkpoint_state)} parametri dal checkpoint")
    
    # 4. Freezal parti del modello se richiesto
    if freeze_encoder:
        print("Freezing encoder layers...")
        # Freezal basal encoder
        for param in model.basal_encoder.parameters():
            param.requires_grad = False
        
        # Freezal pert encoder (opzionale)
        # for param in model.pert_encoder.parameters():
        #     param.requires_grad = False
        
        # Freezal transformer backbone (mantenendo LoRA trainable se presente)
        for name, param in model.transformer_backbone.named_parameters():
            if "lora_" in name:
                param.requires_grad = True  # Mantieni LoRA trainable
                print(f"  Trainable (LoRA): {name}")
            else:
                param.requires_grad = False
                print(f"  Frozen: {name}")
        
        # Verifica quali parametri sono trainable
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parametri trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # 5. Setup data module (esempio semplificato)
    # In pratica, dovresti usare PerturbationDataModule con la tua configurazione
    print("\nNOTA: Setup del data module da implementare con i tuoi dati")
    print(f"Usa: PerturbationDataModule con config da {data_config_path}")
    
    # 6. Setup trainer
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Callbacks personalizzati
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=output_path / "checkpoints",
            filename="st-{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
        ),
    ]
    
    # Logger (WandB o TensorBoard)
    logger = pl.loggers.TensorBoardLogger(
        save_dir=output_path / "logs",
        name="st_training"
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_steps=10000,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        val_check_interval=100,
        log_every_n_steps=10,
    )
    
    print("\n" + "="*80)
    print("SETUP COMPLETATO!")
    print("="*80)
    print(f"Output directory: {output_path}")
    print(f"Modello: {model.__class__.__name__}")
    print(f"Encoder frozen: {freeze_encoder}")
    print(f"LoRA enabled: {use_lora}")
    print("\nPer avviare il training, implementa il data module e chiama:")
    print("  trainer.fit(model, datamodule=data_module)")
    print("="*80)
    
    return model, trainer


def example_custom_modification():
    """
    Esempio di come modificare il modello ST per aggiungere funzionalità custom.
    """
    
    print("\n" + "="*80)
    print("ESEMPIO: MODIFICHE CUSTOM AL MODELLO")
    print("="*80)
    
    # File da modificare per customizzazione:
    modifications = {
        "src/PT-RAG/tx/models/state_transition.py": [
            "Linea ~120-360: Costruzione del modello",
            "Linea ~360-370: Logica per freezare parti",
            "Linea ~450-550: Training step (aggiungi loss custom)",
            "Linea ~550-600: Validation step",
        ],
        "src/PT-RAG/_cli/_tx/_train.py": [
            "Linea ~103-180: Caricamento checkpoint",
            "Linea ~180-250: Setup trainer e callbacks",
        ],
        "src/PT-RAG/tx/callbacks/": [
            "Aggiungi custom callbacks per monitoring",
        ],
    }
    
    for file, changes in modifications.items():
        print(f"\n📁 {file}:")
        for change in changes:
            print(f"   • {change}")
    
    print("\n" + "="*80)
    print("ESEMPIO DI MODIFICA: Aggiungere una loss custom")
    print("="*80)
    
    code_example = '''
# In src/state/tx/models/state_transition.py, nel metodo training_step():

def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, padded=True) -> torch.Tensor:
    # ... codice esistente ...
    
    main_loss = self.loss_fn(pred, target).nanmean()
    self.log("train_loss", main_loss)
    
    # ✨ AGGIUNGI QUI LA TUA LOSS CUSTOM ✨
    # Esempio: Regularizzazione sulla norma delle predizioni
    pred_norm_loss = torch.norm(pred, p=2, dim=-1).mean()
    self.log("train/pred_norm", pred_norm_loss)
    
    # Combina le loss
    total_loss = main_loss + 0.01 * pred_norm_loss  # 0.01 è il peso
    
    return total_loss
'''
    print(code_example)
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Esempio di utilizzo
    print("="*80)
    print("ESEMPIO: TRAINING ST CON SE PRE-ADDESTRATO")
    print("="*80)
    
    # Parametri di esempio (modifica con i tuoi path)
    SE_CHECKPOINT = "/path/to/SE-600M/se600m_epoch15.ckpt"
    DATA_CONFIG = "examples/fewshot.toml"
    OUTPUT_DIR = os.path.expanduser("~/PT-RAG/experiments/st_with_pretrained_se")
    
    print(f"\nConfigurazione:")
    print(f"  SE Checkpoint: {SE_CHECKPOINT}")
    print(f"  Data Config: {DATA_CONFIG}")
    print(f"  Output Dir: {OUTPUT_DIR}")
    print(f"\nNOTA: Modifica questi path nel file prima di eseguire!")
    print("="*80)
    
    # Mostra esempio di modifiche custom
    example_custom_modification()
    
    # Decommenta per eseguire il training (dopo aver configurato i path corretti)
    # model, trainer = load_pretrained_se_and_train_st(
    #     se_checkpoint_path=SE_CHECKPOINT,
    #     data_config_path=DATA_CONFIG,
    #     output_dir=OUTPUT_DIR,
    #     freeze_encoder=True,
    #     use_lora=False,
    # )
    
    print("\n✅ Script completato! Leggi il codice per capire come procedere.")
