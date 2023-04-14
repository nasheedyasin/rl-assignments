import torch
import pytorch_lightning as pl

from torch.nn import functional as F
from transformers import AutoModelForCausalLM


class DialogAgent(pl.LightningModule):
    def __init__(
        self,
        mpath: str,
        lr: float = 3e-5,
        weight_decay: float = 1e-4,
        embedding_size: int = None
    ):
        """
        Args:
            mpath (str): Path to the `transformer` artifact files.
            task_head_lr (float, optional): Task head/s' learning rate.
            Defaults to 2e-4.
            weight_decay (float, optional): Common weight decay co-efficient.
            Defaults to 1e-4.
            embedding_size (int, optional): Length of the tokenizer (len(tokenizer)).
            Use this arg if spl tokens have been added to the tokenizer.
        """
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay

        # Base model        
        self.clm = AutoModelForCausalLM.from_pretrained(
            mpath
        )

        # Resize the embedding layer if needed
        if embedding_size is not None:
            self.clm.resize_token_embeddings(embedding_size)

        # GPT Models don't have a padding requirement, hence this is not set
        # GPT Models have all special tokens set to eos_token
        if not self.clm.config.pad_token_id:
            self.clm.config.pad_token_id = \
                self.clm.config.eos_token_id

        # Save the init arguments
        self.save_hyperparameters()

    def forward(self, text_tokens):
        # Push all inputs to the device in use
        text_tokens = {k: v.to(self.device) for k, v in text_tokens.items()}

        labels = text_tokens.get('labels')

        return self.clm(**text_tokens, labels=labels)

    def common_step(self, batch, batch_idx):
        ids, text_tokens = batch
        output = self(text_tokens)

        return {
            'loss': output.loss
        }

    def training_step(self, batch, batch_idx, *args, **kwargs):
        loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item(), prog_bar=True)

        return loss_dict

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        for k,v in loss_dict.items():
            self.log("val_" + k, v.item(), prog_bar=True)

        return loss_dict

    def test_step(self, batch, batch_idx, *args, **kwargs):
        loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        for k,v in loss_dict.items():
            self.log("test_" + k, v.item(), prog_bar=True)


    def predict_step(self, batch, batch_idx):
        ids, text_tokens = batch

        with torch.no_grad():
            output = self.clm.generate(
                text_tokens['input_ids'],
                max_length=1000
            )

        return ids, output

    def configure_optimizers(self):
        param_dicts = [
            {"params": self.parameters()}
        ]

        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=4),
                "monitor": "val_loss"
            },
        }

