import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from torch.optim import Optimizer


class GradientCheckerCallback(Callback):
    """
    Callback per controllare i gradienti vanishing dopo il backward pass.
    """

    def __init__(self, check_every_n_steps: int = 100):
        """
        Args:
            check_every_n_steps: Controlla i gradienti ogni N step di training
        """
        super().__init__()
        self.check_every_n_steps = check_every_n_steps

    def on_before_optimizer_step(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer: Optimizer
    ) -> None:
        """Chiamato dopo backward() ma prima di optimizer.step()"""
        # Controlla solo ogni N step per non rallentare il training
        if trainer.global_step % self.check_every_n_steps == 0:
            print(f"\n{'='*80}")
            print(f"🔍 Gradient Check at Step {trainer.global_step}")
            print(f"{'='*80}")
            self.check_vanishing_gradients(pl_module)
            print(f"{'='*80}\n")

    @staticmethod
    def check_vanishing_gradients(model):
        """
        Controlla i gradienti per rilevare vanishing gradients.
        """
        gradients = {}
        params_with_grad = 0
        params_without_grad = 0
        no_grad_names = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    gradients[name] = grad_norm
                    if grad_norm < 1e-8:
                        print(f"⚠️  Very small gradient in {name}: {grad_norm:.2e}")
                    params_with_grad += 1
                else:
                    no_grad_names.append(name)
                    gradients[name] = 0.0
                    params_without_grad += 1
        
        print(f"📊 Gradient Summary: {params_with_grad} params with gradients, {params_without_grad} without")
        
        # Check if gene_decoder exists and has no gradients
        gene_decoder_params = [name for name in no_grad_names if 'gene_decoder' in name]
        if gene_decoder_params:
            print(f"ℹ️  gene_decoder has {len(gene_decoder_params)} params with no gradients")
            print(f"   This is NORMAL if output_space != 'gene' or pert_cell_counts not in batch")
        
        # Print which trainable params have no gradient
        if no_grad_names:
            print(f"❌ {len(no_grad_names)} trainable params without gradients:")
            for name in no_grad_names[:10]:  # Print first 10
                print(f"   - {name}")
            if len(no_grad_names) > 10:
                print(f"   ... and {len(no_grad_names) - 10} more")
        
        if gradients:
            max_grad = max(gradients.values())
            min_grad = min([g for g in gradients.values() if g > 0], default=0.0)  # Exclude zeros
            avg_grad = sum(gradients.values()) / len(gradients)
            print(f"📈 Gradient range: min={min_grad:.2e}, max={max_grad:.2e}, avg={avg_grad:.2e}")
        
        return gradients
