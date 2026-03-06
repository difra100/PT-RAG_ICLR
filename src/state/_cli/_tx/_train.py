import argparse as ap
import copy
from omegaconf import DictConfig, OmegaConf
import faiss
import numpy as np

def add_arguments_train(parser: ap.ArgumentParser):
    # Allow remaining args to be passed through to Hydra
    parser.add_argument("hydra_overrides", nargs="*", help="Hydra configuration overrides (e.g., data.batch_size=32)")
    # Add custom help handler
    parser.add_argument("--help", "-h", action="store_true", help="Show configuration help with all parameters")


def run_tx_train(cfg: DictConfig):
    import json
    import logging
    import os
    import pickle
    import shutil
    from os.path import exists, join
    from pathlib import Path

    import lightning.pytorch as pl
    import torch
    from cell_load.data_modules import PerturbationDataModule
    from cell_load.utils.modules import get_datamodule
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.plugins.precision import MixedPrecision

    from ...tx.callbacks import (
        BatchSpeedMonitorCallback,
        CumulativeFLOPSCallback,
        GradNormCallback,
        ModelFLOPSUtilizationCallback,
        WandbLossCallback,
        GradientCheckerCallback,
    )
    from ...tx.utils import get_checkpoint_callbacks, get_lightning_module, get_loggers

    logger = logging.getLogger(__name__)
    torch.set_float32_matmul_precision("medium")

    cfg_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # Setup output directory
    run_output_dir = join(cfg["output_dir"], cfg["name"])
    if os.path.exists(run_output_dir) and cfg["overwrite"]:
        print(f"Output dir {run_output_dir} already exists, overwriting")
        shutil.rmtree(run_output_dir)
    os.makedirs(run_output_dir, exist_ok=True)

    # Set up wandb directory if needed
    if cfg["use_wandb"]:
        os.makedirs(cfg["wandb"]["local_wandb_dir"], exist_ok=True)

    with open(join(run_output_dir, "config.yaml"), "w") as f:
        f.write(cfg_yaml)

    # Set random seeds
    pl.seed_everything(cfg["training"]["train_seed"])

    # if the provided pert_col is drugname_drugconc, hard code the value of control pert
    # this is because it's surprisingly hard to specify a list of tuples in the config as a string
    if cfg["data"]["kwargs"]["pert_col"] == "drugname_drugconc":
        cfg["data"]["kwargs"]["control_pert"] = "[('DMSO_TF', 0.0, 'uM')]"

    # Initialize data module. this is backwards compatible with previous configs
    try:
        sentence_len = cfg["model"]["cell_set_len"]
    except KeyError:
        if cfg["model"]["name"].lower() in ["cpa", "scvi"] or cfg["model"]["name"].lower().startswith("scgpt"):
            if "cell_sentence_len" in cfg["model"]["kwargs"] and cfg["model"]["kwargs"]["cell_sentence_len"] > 1:
                sentence_len = cfg["model"]["kwargs"]["cell_sentence_len"]
                cfg["training"]["batch_size"] = 1
            else:
                sentence_len = 1
        else:
            try:
                sentence_len = cfg["model"]["kwargs"]["transformer_backbone_kwargs"]["n_positions"]
            except:
                sentence_len = cfg["model"]["kwargs"]["transformer_backbone_kwargs"]["max_position_embeddings"]

    if cfg["model"]["name"].lower().startswith("scgpt"):  # scGPT uses log-normalized expression
        cfg["data"]["kwargs"]["transform"] = "log-normalize"
        cfg["data"]["kwargs"]["hvg_names_uns_key"] = (
            "hvg_names" if cfg["data"]["kwargs"]["train_task"] != "replogle" else None
        )  # TODO: better to not hardcode this

        cfg["data"]["kwargs"]["dataset_cls"] = "scGPTPerturbationDataset"

        model_dir = Path(cfg["model"]["kwargs"]["pretrained_path"])

        vocab_file = model_dir / "vocab.json"

        vocab = json.load(open(vocab_file, "r"))
        cfg["model"]["kwargs"]["pad_token_id"] = vocab["<pad>"]
        for s in cfg["model"]["kwargs"]["special_tokens"]:
            if s not in vocab:
                vocab[s] = len(vocab)

        cfg["data"]["kwargs"]["vocab"] = vocab
        cfg["data"]["kwargs"]["perturbation_type"] = cfg["model"]["kwargs"]["perturbation_type"]
        cfg["model"]["kwargs"]["ntoken"] = len(vocab)
        cfg["model"]["kwargs"]["d_model"] = cfg["model"]["kwargs"]["embsize"]

        logger.info("Added vocab and hvg_names_uns_key to data kwargs for scGPT")

    elif cfg["model"]["name"].lower() == "cpa" and cfg["model"]["kwargs"]["recon_loss"] == "gauss":
        cfg["data"]["kwargs"]["transform"] = "log-normalize"
    elif cfg["model"]["name"].lower() == "scvi":
        cfg["data"]["kwargs"]["transform"] = None

    data_module: PerturbationDataModule = get_datamodule(
        cfg["data"]["name"],
        cfg["data"]["kwargs"],
        batch_size=cfg["training"]["batch_size"],
        cell_sentence_len=sentence_len,
    )

    # Modify pert_onehot_map BEFORE setup if using GenePT embeddings
    if cfg['training'].get('use_genept', False):
        
        print("Loading GenePT embeddings before data_module setup...")
        file_genept_path = 'genept_emb/genept_emb/GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle'
        with open(file_genept_path, "rb") as f:
            genept_data = pickle.load(f)
        
        # Get embedding dimension from any GenePT embedding
        sample_key = list(genept_data.keys())[0]
        genept_emb_dim = len(genept_data[sample_key])
        
        # Create new pert_onehot_map with GenePT embeddings
        new_pert_onehot_map = {}
        missing_perts = []
        
        for pert_n in list(data_module.pert_onehot_map.keys()):
            if str(pert_n) in genept_data:
                new_pert_onehot_map[pert_n] = torch.tensor(genept_data[str(pert_n)])
            elif str(pert_n) == 'non-targeting':    
                # For missing perturbations (like controls), use zero vector
                missing_perts.append(pert_n)
                new_pert_onehot_map[pert_n] = torch.zeros(genept_emb_dim, dtype=torch.float32)
        
        if missing_perts:
            print(f"⚠️  {len(missing_perts)} perturbations not in GenePT, assigned zero embeddings:")
            for pert in missing_perts[:5]:  # Show first 5
                print(f"     - {pert}")
            if len(missing_perts) > 5:
                print(f"     ... and {len(missing_perts) - 5} more")
        
        # Replace the map BEFORE setup
        data_module.pert_onehot_map = new_pert_onehot_map
        print(f"✅ Replaced pert_onehot_map with GenePT embeddings for {len(new_pert_onehot_map)} perturbations")

    data_module.setup(stage="fit")
    
    
    # Filter datasets to keep only perturbations with GenePT embeddings
    if cfg['training'].get('use_genept', False):
        print("Filtering datasets to keep only perturbations present in GenePT...")
        valid_perts = set(data_module.pert_onehot_map.keys())
        
        def filter_subset_by_perts(subset, valid_perts):
            """Filter a Subset keeping only cells with valid perturbations or controls."""
            underlying_ds = subset.dataset
            cache = underlying_ds.metadata_cache
            control_pert_code = cache.control_pert_code
            original_indices = subset.indices
            valid_mask = []
            for idx in original_indices:
                pert_code = cache.pert_codes[idx]
                if pert_code == control_pert_code:
                    valid_mask.append(True)
                else:
                    pert_name = cache.pert_categories[pert_code]
                    valid_mask.append(pert_name in valid_perts)
            valid_mask = np.array(valid_mask)
            filtered_indices = original_indices[valid_mask]
            subset.indices = filtered_indices
            return len(original_indices), len(filtered_indices)
        
        
        total_removed = 0
        total_kept = 0
        
        for split_name, datasets in [
            ("train", data_module.train_datasets),
            ("val", data_module.val_datasets),
            ("test", data_module.test_datasets)
        ]:
            if not datasets:
                continue
                
            for i, subset in enumerate(datasets):
                original_count, filtered_count = filter_subset_by_perts(subset, valid_perts)
                removed = original_count - filtered_count
                total_removed += removed
                total_kept += filtered_count
                
                if removed > 0:
                    print(f"  {split_name}[{i}]: kept {filtered_count}/{original_count} cells ({removed} removed)")
        
        print(f"✅ Filtering complete:")
        print(f"   - Total cells kept: {total_kept}")
        print(f"   - Total cells removed: {total_removed}")
        print(f"   - Perturbations in GenePT: {len(valid_perts)}")
    
    # Save data_module AFTER setup and filtering to preserve GenePT embeddings
    with open(join(run_output_dir, "data_module.torch"), "wb") as f:
        # TODO-Abhi: only save necessary data
        data_module.save_state(f)
        
    dl = data_module.train_dataloader()
    print("num_workers:", dl.num_workers)
    print("batch size:", dl.batch_size)

    train_loader = data_module.train_dataloader()
    


    
    
    var_dims = data_module.get_var_dims()  # {"gene_dim": …, "hvg_dim": …}
    
    
    
    if cfg["data"]["kwargs"]["output_space"] == "gene":
        gene_dim = var_dims.get("hvg_dim", 2000)  # fallback if key missing
    else:
        gene_dim = var_dims.get("gene_dim", 2000)  # fallback if key missing
    latent_dim = var_dims["output_dim"]  # same as model.output_dim
    hidden_dims = cfg["model"]["kwargs"].get("decoder_hidden_dims", [1024, 1024, 512])

    decoder_cfg = dict(
        latent_dim=latent_dim,
        gene_dim=gene_dim,
        hidden_dims=hidden_dims,
        dropout=cfg["model"]["kwargs"].get("decoder_dropout", 0.1),
        residual_decoder=cfg["model"]["kwargs"].get("residual_decoder", False),
    )

    # tuck it into the kwargs that will reach the LightningModule
    cfg["model"]["kwargs"]["decoder_cfg"] = decoder_cfg

    # Save the onehot maps as pickle files instead of storing in config
    cell_type_onehot_map_path = join(run_output_dir, "cell_type_onehot_map.pkl")
    pert_onehot_map_path = join(run_output_dir, "pert_onehot_map.pt")
    batch_onehot_map_path = join(run_output_dir, "batch_onehot_map.pkl")
    var_dims_path = join(run_output_dir, "var_dims.pkl")

    with open(cell_type_onehot_map_path, "wb") as f:
        pickle.dump(data_module.cell_type_onehot_map, f)
    torch.save(data_module.pert_onehot_map, pert_onehot_map_path)
    with open(batch_onehot_map_path, "wb") as f:
        pickle.dump(data_module.batch_onehot_map, f)
    with open(var_dims_path, "wb") as f:
        pickle.dump(var_dims, f)

    if cfg["model"]["name"].lower() in ["cpa", "scvi"] or cfg["model"]["name"].lower().startswith("scgpt"):
        cfg["model"]["kwargs"]["n_cell_types"] = len(data_module.celltype_onehot_map)
        cfg["model"]["kwargs"]["n_perts"] = len(data_module.pert_onehot_map)
        cfg["model"]["kwargs"]["n_batches"] = len(data_module.batch_onehot_map)

    # Create model
    model = get_lightning_module(
        cfg["model"]["name"],
        cfg["data"]["kwargs"],
        cfg["model"]["kwargs"],
        cfg["training"],
        data_module.get_var_dims(),
    )

    
    
    if cfg["training"]["rag"]:
        import faiss
        
        # Collect train perturbations and build name-to-index mapping
        train_perts = set()
        train_list_tensor = []
        train_name_to_idx = {}  # Maps pert_name -> numerical index in FAISS
        
        
        # Build train perturbation list from the already-modified pert_onehot_map
        for pert_n in list(data_module.pert_onehot_map.keys()):
            if pert_n not in train_perts:
                train_perts.add(pert_n)
                train_name_to_idx[pert_n] = len(train_list_tensor)
                train_list_tensor.append(data_module.pert_onehot_map[pert_n])
        
        train_pert_emb = torch.stack(train_list_tensor)
        
        # Build FAISS index (will be populated during training, saved after)
        embedding_dim = cfg["model"]["kwargs"]["hidden_dim"]
        train_index = faiss.IndexFlatL2(embedding_dim if not cfg['training']['retrieve_than_predict'] else train_pert_emb.shape[1])
        
        # Attach to model for use during training
        # Index will be dynamically updated during forward passes
        setattr(model, "train_perts", train_pert_emb)
        setattr(model, "train_index", train_index)
        setattr(model, "train_name_to_idx", train_name_to_idx)
        if cfg['training']['use_genept']:
            setattr(model, "pert_dim", train_pert_emb.shape[1])  # GenePT embedding dimension
        print(f"✅ RAG setup initialized:")
        print(f"   - Train: {len(train_name_to_idx)} perturbations")
        print(f"   - FAISS index will be populated after training with final encoder weights")
    
    print(
        f"Model created. Estimated params size: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3:.2f} GB"
    )
    
    # Use training config for wandb settings
    use_wandb = cfg["use_wandb"]
    run_name = cfg["name"]
    
    loggers = get_loggers(
        output_dir=cfg["output_dir"],
        name=run_name,
        wandb_project="PT-RAG",
        wandb_entity="name",
        local_wandb_dir=cfg.get("wandb", {}).get("local_wandb_dir", "./wandb"),
        use_wandb=use_wandb,
        cfg=cfg,
    )

    # If using wandb, store the run path in a text file for eval
    # and set up wandb.watch to monitor the model
    wandb_logger = None
    for lg in loggers:
        if isinstance(lg, WandbLogger):
            wandb_logger = lg
            wandb_info_path = os.path.join(run_output_dir, "wandb_path.txt")
            with open(wandb_info_path, "w") as f:
                f.write(lg.experiment.path)
            
            # Watch the model to log gradients and parameters
            if use_wandb:
                import wandb
                wandb.watch(model, log="all", log_freq=cfg["training"].get("log_every_n_steps", 100))
                print(f"✅ wandb.watch enabled: logging gradients and parameters every {cfg['training'].get('log_every_n_steps', 100)} steps")
            break

    # Set up callbacks
    ckpt_callbacks = get_checkpoint_callbacks(
        cfg["output_dir"],
        run_name,
        cfg["training"]["val_freq"],
        cfg["training"].get("ckpt_every_n_steps", 4000),
    )
    # Add BatchSpeedMonitorCallback to log batches per second to wandb
    batch_speed_monitor = BatchSpeedMonitorCallback()

    callbacks = ckpt_callbacks + [batch_speed_monitor]
    
    # Add WandbLossCallback if wandb is enabled
    if use_wandb:
        wandb_loss_callback = WandbLossCallback()
        callbacks.append(wandb_loss_callback)
        print("✅ WandbLossCallback added: will log train/val loss to wandb")
    
    # Add GradientCheckerCallback to check for vanishing gradients
    if cfg["training"].get("check_gradients", False):
        gradient_check_interval = cfg["training"].get("gradient_check_interval", 100)
        gradient_checker = GradientCheckerCallback(check_every_n_steps=gradient_check_interval)
        callbacks.append(gradient_checker)
        print(f"✅ GradientCheckerCallback added: will check gradients every {gradient_check_interval} steps")

    # Track gradient norm only for state transition model
    if cfg["model"]["name"] == "state":
        callbacks.append(GradNormCallback())

    # Add ModelFLOPSUtilizationCallback to track and log MFU. currently only works for state transition model
    if cfg["training"]["use_mfu"] and cfg["model"]["name"] == "state":
        mfu_available_flops = cfg["training"]["mfu_kwargs"]["available_flops"]
        mfu_use_backward = cfg["training"]["mfu_kwargs"]["use_backward"]
        mfu_logging_interval = cfg["training"]["mfu_kwargs"]["logging_interval"]
        mfu_window_size = cfg["training"]["mfu_kwargs"]["window_size"]
        mfu_cb = ModelFLOPSUtilizationCallback(
            available_flops=mfu_available_flops,
            use_backward=mfu_use_backward,
            logging_interval=mfu_logging_interval,
            cell_set_len=cfg["model"]["kwargs"]["cell_set_len"],
            window_size=mfu_window_size,
        )

        callbacks.append(mfu_cb)

        # Add CumulativeFLOPSCallback to track cumulative FLOPs
        cumulative_flops_use_backward = cfg["training"]["cumulative_flops_use_backward"]
        cumulative_flops_cb = CumulativeFLOPSCallback(use_backward=cumulative_flops_use_backward)
        callbacks.append(cumulative_flops_cb)

    logger.info("Loggers and callbacks set up.")

    if cfg["model"]["name"].lower().startswith("scgpt"):
        plugins = [
            MixedPrecision(
                precision="bf16-mixed",
                device="cuda",
            )
        ]
    else:
        plugins = []

    if torch.cuda.is_available():
        accelerator = "gpu"
    else:
        accelerator = "cpu"

    # Decide on trainer params
    trainer_kwargs = dict(
        accelerator=accelerator,
        devices=cfg["training"].get("devices", 1),
        strategy=cfg["training"].get("strategy", "auto"),
        max_steps=cfg["training"]["max_steps"],  # for normal models
        check_val_every_n_epoch=None,
        val_check_interval=cfg["training"]["val_freq"],
        logger=loggers,
        plugins=plugins,
        callbacks=callbacks,
        gradient_clip_val=cfg["training"]["gradient_clip_val"] if cfg["model"]["name"].lower() != "cpa" else None,
        use_distributed_sampler=False,  # Prevent Lightning from wrapping PerturbationBatchSampler with DistributedSampler
    )

    # Align logging cadence with rolling MFU window (and W&B logging)
    if "log_every_n_steps" in cfg["training"]:
        trainer_kwargs["log_every_n_steps"] = cfg["training"]["log_every_n_steps"]


    # Build trainer
    print(f"Building trainer with kwargs: {trainer_kwargs}")
    trainer = pl.Trainer(**trainer_kwargs)
    print("Trainer built successfully")

    # Load checkpoint if exists
    checkpoint_path = join(ckpt_callbacks[0].dirpath, "last.ckpt")
    if not exists(checkpoint_path):
        checkpoint_path = None
    else:
        logging.info(f"!! Resuming training from {checkpoint_path} !!")

    print(f"Model device: {next(model.parameters()).device}")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    logger.info("Starting trainer fit.")

    # if a checkpoint does not exist, start with the provided checkpoint
    # this is mainly used for pretrain -> finetune workflows
    manual_init = cfg["model"]["kwargs"].get("init_from", None)
    if checkpoint_path is None and manual_init is not None:
        print(f"Loading manual checkpoint from {manual_init}")
        checkpoint_path = manual_init
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_state = model.state_dict()
        checkpoint_state = checkpoint["state_dict"]

        # Check if output_space differs between current config and checkpoint
        checkpoint_output_space = checkpoint.get("hyper_parameters", {}).get("output_space", "gene")
        current_output_space = cfg["data"]["kwargs"]["output_space"]

        if checkpoint_output_space != current_output_space:
            print(
                f"Output space mismatch: checkpoint has '{checkpoint_output_space}', current config has '{current_output_space}'"
            )
            print("Creating new decoder for the specified output space...")

            if cfg["model"]["kwargs"].get("gene_decoder_bool", True) == False:
                model._decoder_externally_configured = False
            else:
                # Override the decoder_cfg to match the new output_space
                if current_output_space == "gene":
                    new_gene_dim = var_dims.get("hvg_dim", 2000)
                else:  # output_space == "all"
                    new_gene_dim = var_dims.get("gene_dim", 2000)

                new_decoder_cfg = dict(
                    latent_dim=var_dims["output_dim"],
                    gene_dim=new_gene_dim,
                    hidden_dims=cfg["model"]["kwargs"].get("decoder_hidden_dims", [1024, 1024, 512]),
                    dropout=cfg["model"]["kwargs"].get("decoder_dropout", 0.1),
                    residual_decoder=cfg["model"]["kwargs"].get("residual_decoder", False),
                )

                # Update the model's decoder_cfg and rebuild decoder
                model.decoder_cfg = new_decoder_cfg
                model._build_decoder()
                model._decoder_externally_configured = True  # Mark that decoder was configured externally
                print(f"Created new decoder for output_space='{current_output_space}' with gene_dim={new_gene_dim}")

        pert_encoder_weight_key = "pert_encoder.0.weight"
        if pert_encoder_weight_key in checkpoint_state:
            checkpoint_pert_dim = checkpoint_state[pert_encoder_weight_key].shape[1]
            if checkpoint_pert_dim != model.pert_dim:
                print(
                    f"pert_encoder input dimension mismatch: model.pert_dim = {model.pert_dim} but checkpoint expects {checkpoint_pert_dim}. Overriding model's pert_dim and rebuilding pert_encoder."
                )
                # Rebuild the pert_encoder with the new pert input dimension
                from ...tx.models.utils import build_mlp

                model.pert_encoder = build_mlp(
                    in_dim=model.pert_dim,
                    out_dim=model.hidden_dim,
                    hidden_dim=model.hidden_dim,
                    n_layers=model.n_encoder_layers,
                    dropout=model.dropout,
                    activation=model.activation_class,
                )
            else:
                print("WARNING: pert_encoder will not be rebuilt since input dimension matches")

        # Filter out mismatched size parameters
        filtered_state = {}
        for name, param in checkpoint_state.items():
            if name in model_state:
                if param.shape == model_state[name].shape:
                    filtered_state[name] = param
                else:
                    print(
                        f"Skipping parameter {name} due to shape mismatch: checkpoint={param.shape}, model={model_state[name].shape}"
                    )
            else:
                print(f"Skipping parameter {name} as it doesn't exist in the current model")

        # Load the filtered state dict
        model.load_state_dict(filtered_state, strict=False)
        print("About to call trainer.fit() with manual checkpoint...")

        # Train - for clarity we pass None
        trainer.fit(
            model,
            datamodule=data_module,
            ckpt_path=None,
        )
        print("trainer.fit() completed with manual checkpoint")
    else:
        print(f"About to call trainer.fit() with checkpoint_path={checkpoint_path}")
        # Train
        trainer.fit(
            model,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )
        print("trainer.fit() completed")

    print("Training completed, saving final checkpoint...")
    train_name_to_idx = getattr(model, "train_name_to_idx", None)
    train_mapping_path = join(run_output_dir, "train_name_to_idx.pkl")
    with open(train_mapping_path, "wb") as f:
        pickle.dump(train_name_to_idx, f)
    # at this point if checkpoint_path does not exist, manually create one
    checkpoint_path = join(ckpt_callbacks[0].dirpath, "final.ckpt")
    if not exists(checkpoint_path):
        trainer.save_checkpoint(checkpoint_path)
    
    if cfg["training"]["rag"] and not cfg['training']['retrieve_than_predict']:
        print("Populating FAISS index with final trained encoder weights...")
        
        # Get the perturbations and mappings from the model
        train_index = getattr(model, "train_index", None)
        train_name_to_idx = getattr(model, "train_name_to_idx", None)
        train_perts = getattr(model, "train_perts", None)
        
        if train_index is not None and train_perts is not None:
            # Encode perturbations with the FINAL trained encoder
            device = next(model.parameters()).device
            model.eval()  # Set to eval mode for consistent encoding
            with torch.no_grad():
                encoded_train_perts = model.encode_perturbation(train_perts.to(device))
                # Normalize for cosine similarity
                encoded_train_perts_np = encoded_train_perts.cpu().numpy()
                faiss.normalize_L2(encoded_train_perts_np)
                # Populate the index with final embeddings
                train_index.reset()  # Clear any previous entries
                train_index.add(encoded_train_perts_np)
            
            print(f"✅ FAISS index populated with trained embeddings:")
            print(f"   - Index size: {train_index.ntotal} vectors")
            
            # Save mappings and indices to disk
            train_index_path = join(run_output_dir, "train_faiss.index")
            train_mapping_path = join(run_output_dir, "train_name_to_idx.pkl")
            faiss.write_index(train_index, train_index_path)
            with open(train_mapping_path, "wb") as f:
                pickle.dump(train_name_to_idx, f)
            print(f"   - Saved to: {train_index_path}")
        else:
            print("⚠️ Warning: Could not populate FAISS index - missing model attributes")
