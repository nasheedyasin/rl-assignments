import torch
import pytorch_lightning as pl

from transformers import AutoModelForCausalLM
from torch.nn import functional as F
from sentence_transformers.models import Pooling
from torchmetrics.classification import MulticlassF1Score


class LanguageModel(pl.LightningModule):
    def __init__(
        self,
        mpath: str,
        lr: float = 3e-5,
        weight_decay: float = 1e-4
    ):
        """
        Args:
            mpath (str): Path to the `transformer` artifact files.
            task_head_lr (float, optional): Task head/s' learning rate.
            Defaults to 2e-4.
            weight_decay (float, optional): Common weight decay co-efficient.
            Defaults to 1e-4.
        """
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay

        # Base model        
        self.lang_model = AutoModelForCausalLM.from_pretrained(
            mpath
        )

        # GPT Models don't have a padding requirement, hence this is not set
        # GPT Models have all special tokens set to eos_token
        if not self.lang_model.config.pad_token_id:
            self.lang_model.config.pad_token_id = \
                self.lang_model.config.eos_token_id

        # Save the init arguments
        self.save_hyperparameters()

    def forward(self, text_tokens, labels=None):
        # Push all inputs to the device in use
        text_tokens = {k: v.to(self.device) for k, v in text_tokens.items()}

        if labels is not None:
            labels = labels.to(self.device)

        return self.lang_model(**text_tokens, labels=labels)

    def common_step(self, batch, batch_idx):
        ids, text_tokens, labels = batch
        output = self(text_tokens, labels)

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
            output = self.lang_model.generate(
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

