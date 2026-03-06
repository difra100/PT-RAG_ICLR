import logging
from typing import Dict, Optional
import faiss
import anndata as ad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from geomloss import SamplesLoss
from typing import Tuple

from .base import PerturbationModel
from .decoders import FinetuneVCICountsDecoder
from .decoders_nb import NBDecoder, nb_nll
from .utils import build_mlp, get_activation_class, get_transformer_backbone, apply_lora, cross_attention, get_k_closest_embeddings


logger = logging.getLogger(__name__)


class CombinedLoss(nn.Module):
    """
    Combined Sinkhorn + Energy loss
    """

    def __init__(self, sinkhorn_weight=0.001, energy_weight=1.0, blur=0.05):
        super().__init__()
        self.sinkhorn_weight = sinkhorn_weight
        self.energy_weight = energy_weight
        self.sinkhorn_loss = SamplesLoss(loss="sinkhorn", blur=blur)
        self.energy_loss = SamplesLoss(loss="energy", blur=blur)

    def forward(self, pred, target):
        sinkhorn_val = self.sinkhorn_loss(pred, target)
        energy_val = self.energy_loss(pred, target)
        return self.sinkhorn_weight * sinkhorn_val + self.energy_weight * energy_val


class ConfidenceToken(nn.Module):
    """
    Learnable confidence token that gets appended to the input sequence
    and learns to predict the expected loss value.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.confidence_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.confidence_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.ReLU(),
        )

    def append_confidence_token(self, seq_input: torch.Tensor) -> torch.Tensor:
        """
        Append confidence token to the sequence input.

        Args:
            seq_input: Input tensor of shape [B, S, E]

        Returns:
            Extended tensor of shape [B, S+1, E]
        """
        batch_size = seq_input.size(0)
        confidence_tokens = self.confidence_token.expand(batch_size, -1, -1)
        return torch.cat([seq_input, confidence_tokens], dim=1)

    def extract_confidence_prediction(self, transformer_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract main output and confidence prediction from transformer output.

        Args:
            transformer_output: Output tensor of shape [B, S+1, E]

        Returns:
            main_output: Tensor of shape [B, S, E]
            confidence_pred: Tensor of shape [B, 1]
        """
        main_output = transformer_output[:, :-1, :]  # [B, S, E]
        confidence_output = transformer_output[:, -1:, :]  # [B, 1, E]
        confidence_pred = self.confidence_projection(confidence_output).squeeze(-1)  # [B, 1]

        return main_output, confidence_pred



class StateTransitionPerturbationModel(PerturbationModel):
    """
    This model:
      1) Projects basal expression and perturbation encodings into a shared latent space.
      2) Uses an OT-based distributional loss (energy, sinkhorn, etc.) from geomloss.
      3) Enables cells to attend to one another, learning a set-to-set function rather than
      a sample-to-sample single-cell map.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pert_dim: int,
        batch_dim: int = None,
        basal_mapping_strategy: str = "random",
        predict_residual: bool = True,
        distributional_loss: str = "energy",
        transformer_backbone_key: str = "GPT2",
        transformer_backbone_kwargs: dict = None,
        output_space: str = "gene",
        gene_dim: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            input_dim: dimension of the input expression (e.g. number of genes or embedding dimension).
            hidden_dim: not necessarily used, but required by PerturbationModel signature.
            output_dim: dimension of the output space (genes or latent).
            pert_dim: dimension of perturbation embedding.
            gpt: e.g. "TranslationTransformerSamplesModel".
            model_kwargs: dictionary passed to that model's constructor.
            loss: choice of distributional metric ("sinkhorn", "energy", etc.).
            **kwargs: anything else to pass up to PerturbationModel or not used.
        """
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            gene_dim=gene_dim,
            output_dim=output_dim,
            pert_dim=pert_dim,
            batch_dim=batch_dim,
            output_space=output_space,
            **kwargs,
        )

        self.predict_residual = predict_residual
        self.output_space = output_space
        self.n_encoder_layers = kwargs.get("n_encoder_layers", 2)
        self.n_decoder_layers = kwargs.get("n_decoder_layers", 2)
        self.activation_class = get_activation_class(kwargs.get("activation", "gelu"))
        self.cell_sentence_len = kwargs.get("cell_set_len", 256)
        self.decoder_loss_weight = kwargs.get("decoder_weight", 1.0)
        self.regularization = kwargs.get("regularization", 0.0)
        self.detach_decoder = kwargs.get("detach_decoder", False)
        
        self.rag_mode = kwargs.get("rag", False)
        self.topk_rag = kwargs.get("topk_rag", 5)
        self.rag_weight = kwargs.get("rag_weight", 0.1)
        self.differentiable_rag = kwargs.get("differentiable_rag", False)
        self.gumbel_sparsity_loss = kwargs.get("gumbel_sparsity_loss", False)
        self.gumbel_sparsity_weight = kwargs.get("gumbel_sparsity_weight", 0.01)
        self.transformer_backbone_key = transformer_backbone_key
        self.transformer_backbone_kwargs = transformer_backbone_kwargs
        self.transformer_backbone_kwargs["n_positions"] = self.cell_sentence_len + kwargs.get("extra_tokens", 0)

        self.distributional_loss = distributional_loss
        self.gene_dim = gene_dim

        blur = kwargs.get("blur", 0.05)
        loss_name = kwargs.get("loss", "energy")
        if loss_name == "energy":
            self.loss_fn = SamplesLoss(loss=self.distributional_loss, blur=blur)
        elif loss_name == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_name == "se":
            sinkhorn_weight = kwargs.get("sinkhorn_weight", 0.01)  # 1/100 = 0.01
            energy_weight = kwargs.get("energy_weight", 1.0)
            self.loss_fn = CombinedLoss(sinkhorn_weight=sinkhorn_weight, energy_weight=energy_weight, blur=blur)
        elif loss_name == "sinkhorn":
            self.loss_fn = SamplesLoss(loss="sinkhorn", blur=blur)
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

        self.use_basal_projection = kwargs.get("use_basal_projection", True)
        self._build_networks(lora_cfg=kwargs.get("lora", None))

        
        if self.rag_mode: 
            self.retrieve_than_predict = kwargs.get('retrieve_than_predict', True)
            
            if self.retrieve_than_predict:
                
                
                if not self.differentiable_rag:
                    
                    self.query_enc = build_mlp(
                        in_dim=self.hidden_dim,
                        out_dim=self.hidden_dim,
                        hidden_dim=self.hidden_dim,
                        n_layers=1,
                        dropout=self.dropout,
                        activation=self.activation_class,
                    )
                    self.key_enc = build_mlp(
                        in_dim=self.hidden_dim,
                        out_dim=self.hidden_dim,
                        hidden_dim=self.hidden_dim,
                        n_layers=1,
                        dropout=self.dropout,
                        activation=self.activation_class,
                    )
                    self.value_enc = build_mlp(
                        in_dim=self.hidden_dim,
                        out_dim=self.hidden_dim,
                        hidden_dim=self.hidden_dim,
                        n_layers=1,
                        dropout=self.dropout,
                        activation=self.activation_class,
                    )
                
                if kwargs.get('use_genept', True):
                    
                    if self.differentiable_rag:
                        logger.info("RAG mode with differentiable retrieve-then-predict enabled.")
                        self.context_input_norm = nn.LayerNorm(3 * self.hidden_dim)
                        self.final_project_genept_emb = build_mlp(
                            in_dim=3 * self.hidden_dim,
                            out_dim=self.hidden_dim,
                            hidden_dim=self.hidden_dim,
                            n_layers=1,
                            dropout=self.dropout,
                            activation=self.activation_class,
                        )
                        self.cell_pert_scorer = build_mlp(
                            in_dim=3 * self.hidden_dim,
                            out_dim=2,
                            hidden_dim=self.hidden_dim,
                            n_layers=2,
                            dropout=self.dropout,
                            activation=self.activation_class,
                        )
                    else:
                        logger.info("RAG mode with retrieve-then-predict enabled.")
                        self.project_genept_emb = build_mlp(
                            in_dim=pert_dim,
                            out_dim=self.hidden_dim,
                            hidden_dim=self.hidden_dim,
                            n_layers=1,
                            dropout=self.dropout,
                            activation=self.activation_class,
                        )
            else:
                self.query_enc = build_mlp(
                        in_dim=self.hidden_dim,
                        out_dim=self.hidden_dim,
                        hidden_dim=self.hidden_dim,
                        n_layers=1,
                        dropout=self.dropout,
                        activation=self.activation_class,
                    )
                self.key_enc = build_mlp(
                    in_dim=self.hidden_dim,
                    out_dim=self.hidden_dim,
                    hidden_dim=self.hidden_dim,
                    n_layers=1,
                    dropout=self.dropout,
                    activation=self.activation_class,
                )
                self.value_enc = build_mlp(
                    in_dim=self.hidden_dim,
                    out_dim=self.hidden_dim,
                    hidden_dim=self.hidden_dim,
                    n_layers=1,
                    dropout=self.dropout,
                    activation=self.activation_class,
                )      
                    
                    

                  
                
        
        # Add an optional encoder that introduces a batch variable
        self.batch_encoder = None
        self.batch_dim = None
        self.predict_mean = kwargs.get("predict_mean", False)
        if kwargs.get("batch_encoder", False) and batch_dim is not None:
            self.batch_encoder = nn.Embedding(
                num_embeddings=batch_dim,
                embedding_dim=hidden_dim,
            )
            self.batch_dim = batch_dim

        # if the model is outputting to counts space, apply relu
        # otherwise its in embedding space and we don't want to
        is_gene_space = kwargs["embed_key"] == "X_hvg" or kwargs["embed_key"] is None
        if is_gene_space or self.gene_decoder is None:
            self.relu = torch.nn.ReLU()

        self.use_batch_token = kwargs.get("use_batch_token", False)
        self.basal_mapping_strategy = basal_mapping_strategy
        # Disable batch token only for truly incompatible cases
        disable_reasons = []
        if self.batch_encoder and self.use_batch_token:
            disable_reasons.append("batch encoder is used")
        if basal_mapping_strategy == "random" and self.use_batch_token:
            disable_reasons.append("basal mapping strategy is random")

        if disable_reasons:
            self.use_batch_token = False
            logger.warning(
                f"Batch token is not supported when {' or '.join(disable_reasons)}, setting use_batch_token to False"
            )
            try:
                self.hparams["use_batch_token"] = False
            except Exception:
                pass

        self.batch_token_weight = kwargs.get("batch_token_weight", 0.1)
        self.batch_token_num_classes: Optional[int] = batch_dim if self.use_batch_token else None

        if self.use_batch_token:
            if self.batch_token_num_classes is None:
                raise ValueError("batch_token_num_classes must be set when use_batch_token is True")
            self.batch_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
            self.batch_classifier = build_mlp(
                in_dim=self.hidden_dim,
                out_dim=self.batch_token_num_classes,
                hidden_dim=self.hidden_dim,
                n_layers=1,
                dropout=self.dropout,
                activation=self.activation_class,
            )
        else:
            self.batch_token = None
            self.batch_classifier = None

        self._batch_token_cache: Optional[torch.Tensor] = None
        self._gumbel_weights_cache: Optional[torch.Tensor] = None

        self.confidence_token = None
        self.confidence_loss_fn = None
        if kwargs.get("confidence_token", False):
            self.confidence_token = ConfidenceToken(hidden_dim=self.hidden_dim, dropout=self.dropout)
            self.confidence_loss_fn = nn.MSELoss()

        # Backward-compat: accept legacy key `freeze_pert`
        self.freeze_pert_backbone = kwargs.get("freeze_pert_backbone", kwargs.get("freeze_pert", False))
        if self.freeze_pert_backbone:
            for name, param in self.transformer_backbone.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            for param in self.project_out.parameters():
                param.requires_grad = False

        if kwargs.get("nb_decoder", False):
            self.gene_decoder = NBDecoder(
                latent_dim=self.output_dim + (self.batch_dim or 0),
                gene_dim=gene_dim,
                hidden_dims=[512, 512, 512],
                dropout=self.dropout,
            )

        control_pert = kwargs.get("control_pert", "non-targeting")
        if kwargs.get("finetune_vci_decoder", False):  # TODO: This will go very soon
            gene_names = []

            if output_space == "gene":
                # hvg's but for which dataset?
                if "DMSO_TF" in control_pert:
                    gene_names = np.load(
                        "/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_to_2k_names.npy", allow_pickle=True
                    )
                elif "non-targeting" in control_pert:
                    temp = ad.read_h5ad("/large_storage/ctc/userspace/aadduri/datasets/hvg/replogle/jurkat.h5")
                    # gene_names = temp.var.index.values
            else:
                assert output_space == "all"
                if "DMSO_TF" in control_pert:
                    gene_names = np.load(
                        "/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_names.npy", allow_pickle=True
                    )
                elif "non-targeting" in control_pert:
                    # temp = ad.read_h5ad('/scratch/ctc/ML/vci/paper_replogle/jurkat.h5')
                    # gene_names = temp.var.index.values
                    temp = ad.read_h5ad("/large_storage/ctc/userspace/aadduri/cross_dataset/replogle/jurkat.h5")
                    gene_names = temp.var.index.values

            self.gene_decoder = FinetuneVCICountsDecoder(
                genes=gene_names,
                # latent_dim=self.output_dim + (self.batch_dim or 0),
            )
        print(self)

    def _build_networks(self, lora_cfg=None):
        """
        Here we instantiate the actual GPT2-based model.
        """
        self.pert_encoder = build_mlp(
            in_dim=self.pert_dim,
            out_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_encoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )

        if self.use_basal_projection:
            self.basal_encoder = build_mlp(
                in_dim=self.input_dim,
                out_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_encoder_layers,
                dropout=self.dropout,
                activation=self.activation_class,
            )
        else:
            self.basal_encoder = nn.Linear(self.input_dim, self.hidden_dim)

        self.transformer_backbone, self.transformer_model_dim = get_transformer_backbone(
            self.transformer_backbone_key,
            self.transformer_backbone_kwargs,
        )

        if lora_cfg and lora_cfg.get("enable", False):
            self.transformer_backbone = apply_lora(
                self.transformer_backbone,
                self.transformer_backbone_key,
                lora_cfg,
            )

        self.project_out = build_mlp(
            in_dim=self.hidden_dim,
            out_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_decoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )

        if self.output_space == "all":
            self.final_down_then_up = nn.Sequential(
                nn.Linear(self.output_dim, self.output_dim // 8),
                nn.GELU(),
                nn.Linear(self.output_dim // 8, self.output_dim),
            )

    def encode_perturbation(self, pert: torch.Tensor) -> torch.Tensor:
        """If needed, define how we embed the raw perturbation input."""
        return self.pert_encoder(pert)

    def encode_basal_expression(self, expr: torch.Tensor) -> torch.Tensor:
        """Define how we embed basal state input, if needed."""
        return self.basal_encoder(expr)

    def refine_combined_input(self, combined: torch.Tensor, pert_embedding: torch.Tensor, batch: dict = None) -> torch.Tensor:
        
        if self.rag_mode: 
            
            # Determine which index and encoded perturbations to use
            if hasattr(self, "train_index") and self.train_index is not None:
                # Re-encode with current weights each step since encoder parameters change during training
                with torch.no_grad():
                    encoded_train_perts = self.encode_perturbation(
                        self.train_perts.to(pert_embedding.device)
                    )
                    encoded_train_perts_np = encoded_train_perts.cpu().numpy()
                    faiss.normalize_L2(encoded_train_perts_np)
                    self.train_index.reset()
                    self.train_index.add(encoded_train_perts_np)
                index = self.train_index
                self._encoded_train_perts = encoded_train_perts
            elif hasattr(self, "train_index_final") and self.train_index_final is not None:
                index = self.train_index_final
                if not hasattr(self, "_encoded_train_perts") or self._encoded_train_perts is None:
                    with torch.no_grad():
                        self._encoded_train_perts = torch.from_numpy(
                            self.train_index_final.reconstruct_n(0, self.train_index_final.ntotal)
                        ).to(pert_embedding.device)
            else:
                return combined
            
            encoded_query = pert_embedding  # [B, S, hidden_dim]
            B, S, hidden_dim = encoded_query.shape
            query_flat = encoded_query.reshape(B * S, hidden_dim).cpu().detach().numpy()
            faiss.normalize_L2(query_flat)
            pert_names = batch.get("pert_name", None) if batch is not None else None
            k_search = self.topk_rag + 1
            distances, indices = index.search(query_flat, k_search)
            if pert_names is not None and hasattr(self, "train_name_to_idx"):
                filtered_indices = []
                for batch_idx in range(B * S):
                    sample_idx = batch_idx // S
                    if sample_idx < len(pert_names):
                        query_pert_name = pert_names[sample_idx]
                        if isinstance(query_pert_name, (list, tuple)):
                            query_pert_name = query_pert_name[batch_idx % S] if batch_idx % S < len(query_pert_name) else query_pert_name[0]
                        query_pert_name = query_pert_name if isinstance(query_pert_name, str) else str(query_pert_name)
                        self_idx = self.train_name_to_idx.get(query_pert_name, -1)
                        sample_indices = indices[batch_idx].tolist()
                        filtered = [idx for idx in sample_indices if idx != self_idx][:self.topk_rag]
                        while len(filtered) < self.topk_rag:
                            filtered.append(filtered[-1] if filtered else 0)
                        filtered_indices.append(filtered)
                    else:
                        filtered_indices.append(indices[batch_idx][:self.topk_rag].tolist())
                indices = np.array(filtered_indices)
            else:
                indices = indices[:, :self.topk_rag]
            
            indices_tensor = torch.from_numpy(indices).long().to(pert_embedding.device)
            closest_embs = self._encoded_train_perts[indices_tensor]
            closest_embs = closest_embs.reshape(B, S, self.topk_rag, hidden_dim)
            key_closest_embs = self.key_enc(closest_embs)
            value_closest_embs = self.value_enc(closest_embs)
            combined = self.query_enc(combined)
            combined_input = cross_attention(
                query=combined,
                key=key_closest_embs,
                value=value_closest_embs,
            )[0]
            if self.rag_weight > 0.0:
                combined_input = combined + self.rag_weight * combined_input
            return combined_input

        else: 
            return combined
        
    
    def refine_combined_input_retrieve_than_predict(self, combined: torch.Tensor, pert: torch.Tensor, batch: dict = None) -> torch.Tensor:
        
        if self.rag_mode:
            if hasattr(self, "train_index") and self.train_index is not None:
                encoded_train_perts = self.train_perts.to(pert.device)
                self._encoded_train_perts = encoded_train_perts
            else:
                return combined

            encoded_query = pert  # [B, S, genept_dim]
            B, S, genept_dim = encoded_query.shape
            query_flat = encoded_query.reshape(B * S, genept_dim)
            query_norm = F.normalize(query_flat, p=2, dim=1)
            db_norm = F.normalize(self._encoded_train_perts, p=2, dim=1)
            similarities = torch.matmul(query_norm, db_norm.t())
            pert_names = batch.get("pert_name", None) if batch is not None else None
            if pert_names is not None and hasattr(self, "train_name_to_idx"):
                for batch_idx in range(B * S):
                    sample_idx = batch_idx // S
                    if sample_idx < len(pert_names):
                        query_pert_name = pert_names[sample_idx]
                        if isinstance(query_pert_name, (list, tuple)):
                            query_pert_name = query_pert_name[batch_idx % S] if batch_idx % S < len(query_pert_name) else query_pert_name[0]
                        query_pert_name = query_pert_name if isinstance(query_pert_name, str) else str(query_pert_name)
                        self_idx = self.train_name_to_idx.get(query_pert_name, -1)
                        if self_idx >= 0:
                            similarities[batch_idx, self_idx] = float('-inf')
            _, indices_tensor = torch.topk(similarities, k=self.topk_rag, dim=1, largest=True)
            closest_embs = self._encoded_train_perts[indices_tensor]
            emb_dim = self._encoded_train_perts.shape[-1]
            closest_embs = closest_embs.reshape(B, S, self.topk_rag, emb_dim)
            closest_embs = self.project_genept_emb(closest_embs)
            key_closest_embs = self.key_enc(closest_embs)
            value_closest_embs = self.value_enc(closest_embs)
            combined = self.query_enc(combined)
            combined_input = cross_attention(
                query=combined,
                key=key_closest_embs,
                value=value_closest_embs,
            )[0]
            return combined_input
        else:
            return combined
    
    def refine_combined_input_diff_retrieve_than_predict(self, combined_input, pert_embedding: torch.Tensor, pert: torch.Tensor, cells: torch.Tensor, batch: dict = None) -> torch.Tensor:
        
        if self.rag_mode:
            if hasattr(self, "train_perts") and self.train_perts is not None:
                encoded_train_perts = self.train_perts.to(pert.device)
                self._encoded_train_perts = encoded_train_perts
            else:
                return combined_input
            
            encoded_query = pert
            B, S, genept_dim = encoded_query.shape
            query_flat = encoded_query.reshape(B * S, genept_dim)
            query_norm = F.normalize(query_flat, p=2, dim=1)
            db_norm = F.normalize(self._encoded_train_perts, p=2, dim=1)
            similarities = torch.matmul(query_norm, db_norm.t())
            pert_names = batch.get("pert_name", None) if batch is not None else None
            if pert_names is not None and hasattr(self, "train_name_to_idx"):
                for batch_idx in range(B * S):
                    sample_idx = batch_idx // S
                    if sample_idx < len(pert_names):
                        query_pert_name = pert_names[sample_idx]
                        if isinstance(query_pert_name, (list, tuple)):
                            query_pert_name = query_pert_name[batch_idx % S] if batch_idx % S < len(query_pert_name) else query_pert_name[0]
                        query_pert_name = query_pert_name if isinstance(query_pert_name, str) else str(query_pert_name)
                        self_idx = self.train_name_to_idx.get(query_pert_name, -1)
                        if self_idx >= 0:
                            similarities[batch_idx, self_idx] = float('-inf')
            _, indices_tensor = torch.topk(similarities, k=self.topk_rag, dim=1, largest=True)
            closest_embs = self._encoded_train_perts[indices_tensor]
            emb_dim = self._encoded_train_perts.shape[-1]
            closest_embs = closest_embs.reshape(B, S, self.topk_rag, emb_dim)
            closest_embs = self.encode_perturbation(closest_embs)
            context_input_repr = torch.cat([cells.unsqueeze(2).expand(-1, -1, self.topk_rag, -1), pert_embedding.unsqueeze(2).expand(-1, -1, self.topk_rag, -1), closest_embs], dim=-1)
            # Normalize to prevent extreme scores
            context_input_repr = self.context_input_norm(context_input_repr)
            context_input_scores = self.cell_pert_scorer(context_input_repr)
            # Scale to prevent saturation
            context_input_scores = context_input_scores / context_input_scores.std().clamp(min=1e-6)
            context_input_weights = F.gumbel_softmax(context_input_scores, tau=0.5, dim=-1, hard=True)
            context_input_weights = context_input_weights[..., -1][..., None]
            # Cache weights for sparsity regularization loss (no detach: gradients flow to cell_pert_scorer)
            if self.training and self.gumbel_sparsity_loss:
                self._gumbel_weights_cache = context_input_weights
            context_input_embs = self.final_project_genept_emb(context_input_repr)
            final_context_embs = (context_input_weights * context_input_embs).sum(dim=2)
            return final_context_embs
        else:
            return combined_input
    
    
    def refine_combined_input_diff_retrieve_than_predict_collect_retrieved_perts(self, combined_input, pert_embedding: torch.Tensor, pert: torch.Tensor, cells: torch.Tensor, batch: dict = None) -> Tuple[torch.Tensor, dict]:
        
        retrieval_info = {}
        if self.rag_mode:
            if hasattr(self, "train_perts") and self.train_perts is not None:
                encoded_train_perts = self.train_perts.to(pert.device)
                self._encoded_train_perts = encoded_train_perts
            else:
                return combined_input, retrieval_info
            
            encoded_query = pert
            B, S, genept_dim = encoded_query.shape
            query_flat = encoded_query.reshape(B * S, genept_dim)
            query_norm = F.normalize(query_flat, p=2, dim=1)
            db_norm = F.normalize(self._encoded_train_perts, p=2, dim=1)
            similarities = torch.matmul(query_norm, db_norm.t())
            pert_names = batch.get("pert_name", None) if batch is not None else None
            if pert_names is not None and hasattr(self, "train_name_to_idx"):
                for batch_idx in range(B * S):
                    sample_idx = batch_idx // S
                    if sample_idx < len(pert_names):
                        query_pert_name = pert_names[sample_idx]
                        if isinstance(query_pert_name, (list, tuple)):
                            query_pert_name = query_pert_name[batch_idx % S] if batch_idx % S < len(query_pert_name) else query_pert_name[0]
                        query_pert_name = query_pert_name if isinstance(query_pert_name, str) else str(query_pert_name)
                        self_idx = self.train_name_to_idx.get(query_pert_name, -1)
                        if self_idx >= 0:
                            similarities[batch_idx, self_idx] = float('-inf')
            _, indices_tensor = torch.topk(similarities, k=self.topk_rag, dim=1, largest=True)
            closest_embs = self._encoded_train_perts[indices_tensor]
            emb_dim = self._encoded_train_perts.shape[-1]
            closest_embs = closest_embs.reshape(B, S, self.topk_rag, emb_dim)
            closest_embs = self.encode_perturbation(closest_embs)
            context_input_repr = torch.cat([cells.unsqueeze(2).expand(-1, -1, self.topk_rag, -1), pert_embedding.unsqueeze(2).expand(-1, -1, self.topk_rag, -1), closest_embs], dim=-1)
            # Normalize to prevent extreme scores
            context_input_repr = self.context_input_norm(context_input_repr)
            context_input_scores = self.cell_pert_scorer(context_input_repr)
            # Scale to prevent saturation
            context_input_scores = context_input_scores / context_input_scores.std().clamp(min=1e-6)
            context_input_weights = F.gumbel_softmax(context_input_scores, tau=0.5, dim=-1, hard=True)
            context_input_weights = context_input_weights[..., -1][..., None]
            # Cache weights for sparsity regularization loss (no detach: gradients flow to cell_pert_scorer)
            if self.training and self.gumbel_sparsity_loss:
                self._gumbel_weights_cache = context_input_weights
            context_input_embs = self.final_project_genept_emb(context_input_repr)
            final_context_embs = (context_input_weights * context_input_embs).sum(dim=2)
            if hasattr(self, "train_name_to_idx"):
                train_idx_to_name = {idx: name for name, idx in self.train_name_to_idx.items()}
                cell_types = batch.get("cell_type", None) if batch is not None else None
                retrieval_info["indices"] = indices_tensor.cpu().numpy()
                retrieval_info["weights"] = context_input_weights.squeeze(-1).cpu().numpy()
                retrieval_info["train_idx_to_name"] = train_idx_to_name
                retrieval_info["pert_names"] = pert_names
                retrieval_info["cell_types"] = cell_types
                retrieval_info["B"] = B
                retrieval_info["S"] = S
            return final_context_embs, retrieval_info
        else:
            return combined_input, retrieval_info
    
    
    def forward(self, batch: dict, padded=True, collect_retrieval_info=None) -> torch.Tensor:
        """
        The main forward call. Batch is a flattened sequence of cell sentences,
        which we reshape into sequences of length cell_sentence_len.

        Expects input tensors of shape (B, S, N) where:
        B = batch size
        S = sequence length (cell_sentence_len)
        N = feature dimension

        The `padded` argument here is set to True if the batch is padded. Otherwise, we
        expect a single batch, so that sentences can vary in length across batches.
        """
        if padded:
            pert = batch["pert_emb"].reshape(-1, self.cell_sentence_len, self.pert_dim)
            basal = batch["ctrl_cell_emb"].reshape(-1, self.cell_sentence_len, self.input_dim)
        else:
            # we are inferencing on a single batch, so accept variable length sentences
            pert = batch["pert_emb"].reshape(1, -1, self.pert_dim)
            basal = batch["ctrl_cell_emb"].reshape(1, -1, self.input_dim)

        retrieval_info = {}
        pert_embedding = self.encode_perturbation(pert)
        control_cells = self.encode_basal_expression(basal)
        combined_input = pert_embedding + control_cells
        
        if self.rag_mode:
            if self.retrieve_than_predict:
                if not self.differentiable_rag:
                    combined_input = self.refine_combined_input_retrieve_than_predict(combined_input, pert, batch)
                else:
                    if collect_retrieval_info:
                        combined_input, retrieval_info = self.refine_combined_input_diff_retrieve_than_predict_collect_retrieved_perts(combined_input, pert_embedding, pert, control_cells, batch)
                    else:
                        combined_input = self.refine_combined_input_diff_retrieve_than_predict(combined_input, pert_embedding, pert, control_cells, batch)
            else:
                combined_input = self.refine_combined_input(combined_input, pert_embedding, batch)
        
        
        seq_input = combined_input

        if self.batch_encoder is not None:
            batch_indices = batch["batch"]
            if batch_indices.dim() > 1 and batch_indices.size(-1) == self.batch_dim:
                batch_indices = batch_indices.argmax(-1)
            if padded:
                batch_indices = batch_indices.reshape(-1, self.cell_sentence_len)
            else:
                batch_indices = batch_indices.reshape(1, -1)
            batch_embeddings = self.batch_encoder(batch_indices.long())
            seq_input = seq_input + batch_embeddings

        if self.use_batch_token and self.batch_token is not None:
            batch_size, _, _ = seq_input.shape
            seq_input = torch.cat([self.batch_token.expand(batch_size, -1, -1), seq_input], dim=1)

        confidence_pred = None
        if self.confidence_token is not None:
            seq_input = self.confidence_token.append_confidence_token(seq_input)

        if self.hparams.get("mask_attn", False):
            batch_size, seq_length, _ = seq_input.shape
            device = seq_input.device
            self.transformer_backbone._attn_implementation = "eager"   # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
            base = torch.eye(seq_length, device=device, dtype=torch.bool).view(1, 1, seq_length, seq_length)
            num_heads = self.transformer_backbone.config.num_attention_heads
            attn_mask = base.repeat(batch_size, num_heads, 1, 1)

            outputs = self.transformer_backbone(inputs_embeds=seq_input, attention_mask=attn_mask)
            transformer_output = outputs.last_hidden_state
        else:
            outputs = self.transformer_backbone(inputs_embeds=seq_input)
            transformer_output = outputs.last_hidden_state

        if self.confidence_token is not None and self.use_batch_token and self.batch_token is not None:
            batch_token_pred = transformer_output[:, :1, :]  # [B, 1, H]
            res_pred, confidence_pred = self.confidence_token.extract_confidence_prediction(
                transformer_output[:, 1:, :]
            )
            self._batch_token_cache = batch_token_pred
        elif self.confidence_token is not None:
            res_pred, confidence_pred = self.confidence_token.extract_confidence_prediction(transformer_output)
            self._batch_token_cache = None
        elif self.use_batch_token and self.batch_token is not None:
            batch_token_pred = transformer_output[:, :1, :]  # [B, 1, H]
            res_pred = transformer_output[:, 1:, :]  # [B, S, H]
            self._batch_token_cache = batch_token_pred
        else:
            res_pred = transformer_output
            self._batch_token_cache = None

        if self.predict_residual and self.output_space == "all":
            out_pred = self.project_out(res_pred) + basal
            out_pred = self.final_down_then_up(out_pred)
        elif self.predict_residual:
            out_pred = self.project_out(res_pred + control_cells)
        else:
            out_pred = self.project_out(res_pred)

        is_gene_space = self.hparams["embed_key"] == "X_hvg" or self.hparams["embed_key"] is None
        if is_gene_space or self.gene_decoder is None:
            out_pred = self.relu(out_pred)

        output = out_pred.reshape(-1, self.output_dim)

        if collect_retrieval_info and retrieval_info:
            if confidence_pred is not None:
                return output, confidence_pred, retrieval_info
            else:
                return output, retrieval_info
        elif confidence_pred is not None:
            return output, confidence_pred
        else:
            return output

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, padded=True) -> torch.Tensor:
        confidence_pred = None
        if self.confidence_token is not None:
            pred, confidence_pred = self.forward(batch, padded=padded)
        else:
            pred = self.forward(batch, padded=padded)

        target = batch["pert_cell_emb"]

        if padded:
            pred = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
            target = target.reshape(-1, self.cell_sentence_len, self.output_dim)
        else:
            pred = pred.reshape(1, -1, self.output_dim)
            target = target.reshape(1, -1, self.output_dim)

        main_loss = self.loss_fn(pred, target).nanmean()
        self.log("train_loss", main_loss)

        if hasattr(self.loss_fn, "sinkhorn_loss") and hasattr(self.loss_fn, "energy_loss"):
            sinkhorn_component = self.loss_fn.sinkhorn_loss(pred, target).nanmean()
            energy_component = self.loss_fn.energy_loss(pred, target).nanmean()
            self.log("train/sinkhorn_loss", sinkhorn_component)
            self.log("train/energy_loss", energy_component)

        decoder_loss = None
        total_loss = main_loss

        if self.use_batch_token and self.batch_classifier is not None and self._batch_token_cache is not None:
            logits = self.batch_classifier(self._batch_token_cache)  # [B, 1, C]
            batch_token_targets = batch["batch"]

            B = logits.shape[0]
            C = logits.size(-1)

            # Prepare one label per sequence (all S cells share the same batch)
            if batch_token_targets.dim() > 1 and batch_token_targets.size(-1) == C:
                if padded:
                    target_oh = batch_token_targets.reshape(-1, self.cell_sentence_len, C)
                else:
                    target_oh = batch_token_targets.reshape(1, -1, C)
                sentence_batch_labels = target_oh.argmax(-1)
            else:
                if padded:
                    sentence_batch_labels = batch_token_targets.reshape(-1, self.cell_sentence_len)
                else:
                    sentence_batch_labels = batch_token_targets.reshape(1, -1)

            if sentence_batch_labels.shape[0] != B:
                sentence_batch_labels = sentence_batch_labels.reshape(B, -1)

            if self.basal_mapping_strategy == "batch":
                uniform_mask = sentence_batch_labels.eq(sentence_batch_labels[:, :1]).all(dim=1)
                if not torch.all(uniform_mask):
                    bad_indices = torch.where(~uniform_mask)[0]
                    label_strings = []
                    for idx in bad_indices:
                        labels = sentence_batch_labels[idx].detach().cpu().tolist()
                        logger.error("Batch labels for sentence %d: %s", idx.item(), labels)
                        label_strings.append(f"sentence {idx.item()}: {labels}")
                    raise ValueError(
                        "Expected all cells in a sentence to share the same batch when "
                        "basal_mapping_strategy is 'batch'. "
                        f"Found mixed batch labels: {', '.join(label_strings)}"
                    )

            target_idx = sentence_batch_labels[:, 0]
            if target_idx.numel() != B:
                target_idx = target_idx.reshape(-1)[:B]

            ce_loss = F.cross_entropy(logits.reshape(B, -1, C).squeeze(1), target_idx.long())
            self.log("train/batch_token_loss", ce_loss)
            total_loss = total_loss + self.batch_token_weight * ce_loss

        if self.gene_decoder is not None and "pert_cell_counts" in batch:
            gene_targets = batch["pert_cell_counts"]
            if self.detach_decoder:
                # with some random change, use the true targets
                if np.random.rand() < 0.1:
                    latent_preds = target.reshape_as(pred).detach()
                else:
                    latent_preds = pred.detach()
            else:
                latent_preds = pred

            if isinstance(self.gene_decoder, NBDecoder):
                mu, theta = self.gene_decoder(latent_preds)
                gene_targets = batch["pert_cell_counts"].reshape_as(mu)
                decoder_loss = nb_nll(gene_targets, mu, theta)
            else:
                pert_cell_counts_preds = self.gene_decoder(latent_preds)
                if padded:
                    gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_decoder.gene_dim())
                else:
                    gene_targets = gene_targets.reshape(1, -1, self.gene_decoder.gene_dim())

                decoder_loss = self.loss_fn(pert_cell_counts_preds, gene_targets).mean()

            self.log("decoder_loss", decoder_loss)
            total_loss = total_loss + self.decoder_loss_weight * decoder_loss

        if confidence_pred is not None:
            # Detach main loss to prevent gradients flowing through it
            loss_target = total_loss.detach().clone().unsqueeze(0) * 10
            if confidence_pred.dim() == 2:  # [B, 1]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0), 1)
            else:
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0))
            confidence_loss = self.confidence_loss_fn(confidence_pred.squeeze(), loss_target.squeeze())
            self.log("train/confidence_loss", confidence_loss)
            self.log("train/actual_loss", loss_target.mean())
            confidence_weight = 0.1
            total_loss = total_loss + confidence_weight * confidence_loss
            total_loss = total_loss + confidence_loss

        if self.regularization > 0.0:
            ctrl_cell_emb = batch["ctrl_cell_emb"].reshape_as(pred)
            delta = pred - ctrl_cell_emb
            l1_loss = torch.abs(delta).mean()
            self.log("train/l1_regularization", l1_loss)
            total_loss = total_loss + self.regularization * l1_loss
        
        if self.gumbel_sparsity_loss and self._gumbel_weights_cache is not None:
            sparsity_loss = torch.abs(self._gumbel_weights_cache).mean()
            self.log("train/gumbel_sparsity", sparsity_loss)
            self.log("train/gumbel_active_ratio", (self._gumbel_weights_cache > 0.5).float().mean())
            total_loss = total_loss + self.gumbel_sparsity_weight * sparsity_loss
            self._gumbel_weights_cache = None

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        if self.confidence_token is None:
            pred, confidence_pred = self.forward(batch), None
        else:
            pred, confidence_pred = self.forward(batch)

        pred = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
        target = batch["pert_cell_emb"]
        target = target.reshape(-1, self.cell_sentence_len, self.output_dim)

        loss = self.loss_fn(pred, target).mean()
        self.log("val_loss", loss)

        if hasattr(self.loss_fn, "sinkhorn_loss") and hasattr(self.loss_fn, "energy_loss"):
            sinkhorn_component = self.loss_fn.sinkhorn_loss(pred, target).mean()
            energy_component = self.loss_fn.energy_loss(pred, target).mean()
            self.log("val/sinkhorn_loss", sinkhorn_component)
            self.log("val/energy_loss", energy_component)

        if self.gene_decoder is not None and "pert_cell_counts" in batch:
            gene_targets = batch["pert_cell_counts"]
            latent_preds = pred
            if isinstance(self.gene_decoder, NBDecoder):
                mu, theta = self.gene_decoder(latent_preds)
                gene_targets = batch["pert_cell_counts"].reshape_as(mu)
                decoder_loss = nb_nll(gene_targets, mu, theta)
            else:
                pert_cell_counts_preds = self.gene_decoder(latent_preds).reshape(
                    -1, self.cell_sentence_len, self.gene_decoder.gene_dim()
                )
                gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_decoder.gene_dim())
                decoder_loss = self.loss_fn(pert_cell_counts_preds, gene_targets).mean()
            self.log("val/decoder_loss", decoder_loss)
            loss = loss + self.decoder_loss_weight * decoder_loss

        if confidence_pred is not None:
            # Detach main loss to prevent gradients flowing through it
            loss_target = loss.detach().clone() * 10
            if confidence_pred.dim() == 2:  # [B, 1]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0), 1)
            else:
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0))
            confidence_loss = self.confidence_loss_fn(confidence_pred.squeeze(), loss_target.squeeze())
            self.log("val/confidence_loss", confidence_loss)
            self.log("val/actual_loss", loss_target.mean())

        return {"loss": loss, "predictions": pred}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        if self.confidence_token is None:
            pred, confidence_pred = self.forward(batch, padded=False), None
        else:
            pred, confidence_pred = self.forward(batch, padded=False)

        target = batch["pert_cell_emb"]
        pred = pred.reshape(1, -1, self.output_dim)
        target = target.reshape(1, -1, self.output_dim)
        loss = self.loss_fn(pred, target).mean()
        self.log("test_loss", loss)

        if confidence_pred is not None:
            # Detach main loss to prevent gradients flowing through it
            loss_target = loss.detach().clone() * 10.0
            if confidence_pred.dim() == 2:  # [B, 1]
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0), 1)
            else:
                loss_target = loss_target.unsqueeze(0).expand(confidence_pred.size(0))
            confidence_loss = self.confidence_loss_fn(confidence_pred.squeeze(), loss_target.squeeze())
            self.log("test/confidence_loss", confidence_loss)

    def predict_step(self, batch, batch_idx, padded=True, collect_retrieval_info=False, **kwargs):
        """
        Typically used for final inference. We'll replicate old logic:
         returning 'preds', 'X', 'pert_name', etc.
        
        If collect_retrieval_info=True, also return retrieval information.
        """
        forward_output = self.forward(batch, padded=padded, collect_retrieval_info=collect_retrieval_info)
        if collect_retrieval_info and isinstance(forward_output, tuple) and len(forward_output) == 3:
            # Has retrieval info: (latent_output, confidence_pred, retrieval_info)
            latent_output, confidence_pred, retrieval_info = forward_output
        elif collect_retrieval_info and isinstance(forward_output, tuple) and len(forward_output) == 2:
            # Check if second element is dict (retrieval_info) or tensor (confidence_pred)
            if isinstance(forward_output[1], dict):
                latent_output, retrieval_info = forward_output
                confidence_pred = None
            else:
                latent_output, confidence_pred = forward_output
                retrieval_info = {}
        elif isinstance(forward_output, tuple):
            latent_output, confidence_pred = forward_output
            retrieval_info = {}
        else:
            latent_output = forward_output
            confidence_pred = None
            retrieval_info = {}

        output_dict = {
            "preds": latent_output,
            "pert_cell_emb": batch.get("pert_cell_emb", None),
            "pert_cell_counts": batch.get("pert_cell_counts", None),
            "pert_name": batch.get("pert_name", None),
            "celltype_name": batch.get("cell_type", None),
            "batch": batch.get("batch", None),
            "ctrl_cell_emb": batch.get("ctrl_cell_emb", None),
            "pert_cell_barcode": batch.get("pert_cell_barcode", None),
            "ctrl_cell_barcode": batch.get("ctrl_cell_barcode", None),
        }

        if confidence_pred is not None:
            output_dict["confidence_pred"] = confidence_pred
        if retrieval_info:
            output_dict["retrieval_info"] = retrieval_info

        if self.gene_decoder is not None:
            if isinstance(self.gene_decoder, NBDecoder):
                mu, _ = self.gene_decoder(latent_output)
                pert_cell_counts_preds = mu
            else:
                pert_cell_counts_preds = self.gene_decoder(latent_output)

            output_dict["pert_cell_counts_preds"] = pert_cell_counts_preds

        return output_dict