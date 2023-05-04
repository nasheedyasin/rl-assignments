import torch
import pytorch_lightning as pl

from typing import Dict
from torch.nn import functional as F
from torchmetrics.classification import MultilabelF1Score
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification


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

        return self.clm(**text_tokens)

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

class PersuasionSchemeClassifier(pl.LightningModule):
    def __init__(
        self,
        mpath: str,
        id2label: Dict[int, str], 
        backbone_lr: float = 2e-5,
        task_head_lr: float = 2e-4,
        weight_decay: float = 1e-4,
    ):
        """
        Args:
            mpath (str): Path to the `transformer` artifact files.
            num_labels (int): Number of labels.
            backbone_lr (float, optional): The LLM's learning rate.
            Defaults to 2e-5.
            task_head_lr (float, optional): Task head/s' learning rate.
            Defaults to 2e-4.
            weight_decay (float, optional): Common weight decay co-efficient.
            Defaults to 1e-4.
        """
        super().__init__()
        self.backbone_lr = backbone_lr
        self.task_head_lr = task_head_lr
        self.weight_decay = weight_decay
        self.num_labels = len(id2label)

        # For metric reporting purposes
        self.id2label = id2label

        # Base model        
        self.seq_classifier = AutoModelForSequenceClassification.from_pretrained(
            mpath, num_labels=self.num_labels, problem_type="multi_label_classification"
        )

        # TO-DO: Freeze the backbone model
        # if not backbone_lr:
        #     for param in self.seq_classifier.parameters():
        #         param.requires_grad = False

        # Save the init arguments
        self.save_hyperparameters()

    def forward(self, text_tokens):
        # Push all inputs to the device in use
        text_tokens = {k: v.to(self.device) for k, v in text_tokens.items()}

        return self.seq_classifier(**text_tokens)

    def common_step(self, batch, batch_idx):
        ids, text_tokens = batch
        output = self(text_tokens)

        # Compute the metrics
        granular_metric = MultilabelF1Score(num_labels=self.num_labels, average=None).to(self.device)
        overall_metric = MultilabelF1Score(num_labels=self.num_labels, average='micro').to(self.device)
        macro_metric = MultilabelF1Score(num_labels=self.num_labels).to(self.device)
        reshaped_targets = text_tokens['labels'].to(self.device)
        pred_labels = output.logits.sigmoid()

        return {
            'loss': output.loss,
            'overall_f1_score': overall_metric(
                pred_labels,
                reshaped_targets
            ),
            'macro_f1_score': macro_metric(
                pred_labels,
                reshaped_targets
            ),
            'granular_f1_score': {
                self.id2label[idx]: f1 for idx, f1 in enumerate(granular_metric(
                    pred_labels,
                    reshaped_targets
                ))
            } if self.id2label is not None else granular_metric(
                pred_labels,
                reshaped_targets
            )
        }

    def training_step(self, batch, batch_idx, *args, **kwargs):
        loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        for k, v in loss_dict.items():
            if isinstance(v, dict) or isinstance(v, torch.Tensor): continue
            self.log("train_" + k, v.item(), prog_bar=True)

        return loss_dict

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        for k, v in loss_dict.items():
            if isinstance(v, dict):
                for subk, subv in v.items():
                    self.log(f"val_{k}_{subk}", subv.item(), prog_bar=True)
            elif isinstance(v, torch.Tensor):
                self.log("val_" + k, v, prog_bar=True)
            else:
                self.log("val_" + k, v.item(), prog_bar=True)

        return loss_dict

    def test_step(self, batch, batch_idx, *args, **kwargs):
        loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        for k, v in loss_dict.items():
            if isinstance(v, dict):
                for subk, subv in v.items():
                    self.log(f"test_{k}_{subk}", subv.item(), prog_bar=True)
            elif isinstance(v, torch.Tensor):
                self.log("val_" + k, v, prog_bar=True)
            else:
                self.log("test_" + k, v.item(), prog_bar=True)

    def predict_step(self, batch, batch_idx):
        ids, text_tokens = batch

        with torch.no_grad():
            output = self(text_tokens)
            confidence_scores = output.logits.sigmoid()

        return (
            ids,
            (confidence_scores>0.5).to(torch.float16).cpu(),
            confidence_scores.cpu()
        )

    def configure_optimizers(self):
        param_dicts = [
            {"params": self.parameters()}
            # TO-DO: Implement different LRs for backbone and task head
            # {"params": self.task_head.parameters()},
            # {"params": self.interaction_model.parameters()},
            # {
            #     "params": self.seq_classifier.parameters(),
            #     "lr": self.backbone_lr,
            # },
        ]

        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=self.task_head_lr,
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
