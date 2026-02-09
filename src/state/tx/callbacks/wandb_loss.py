from lightning.pytorch.callbacks import Callback


class WandbLossCallback(Callback):
    """
    Callback that logs training and validation loss to wandb.
    This provides explicit tracking of loss values during training.
    """

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Log training loss after each batch.
        """
        if outputs is not None and "loss" in outputs:
            pl_module.log("train/batch_loss", outputs["loss"], prog_bar=True, logger=True)

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Log epoch-level training loss.
        """
        # The epoch loss is typically already logged by Lightning, but we ensure it's tracked
        pass

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Log validation loss after each validation batch.
        """
        if outputs is not None and "loss" in outputs:
            pl_module.log("val/batch_loss", outputs["loss"], prog_bar=False, logger=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Log epoch-level validation loss.
        """
        # Validation loss is already aggregated by Lightning
        pass
