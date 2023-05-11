import torch
import torch.nn as nn
import pytorch_lightning as pl

from typing import Dict
from collections import OrderedDict
from torch.nn import functional as F
from torchmetrics.classification import MultilabelF1Score
from transformers.activations import NewGELUActivation
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

        # Create the value head network
        hidden_size = self.clm.config.n_embd
        self.value_head = torch.nn.Sequential(OrderedDict([
          ('lin1', torch.nn.Linear(hidden_size, int(hidden_size/4))),
          ('act', NewGELUActivation()), # Stateless, has no params
          ('val_dropout', torch.nn.Dropout(p=.1)),
          ('lin2', torch.nn.Linear(int(hidden_size/4), 1))
        ]))

        # Save the init arguments
        self.save_hyperparameters()

    def forward(self, text_tokens, **kwargs):
        # Push all inputs to the device in use
        text_tokens = {k: v.to(self.device) for k, v in text_tokens.items()}

        return self.clm(**text_tokens, **kwargs)

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


    @torch.no_grad()
    def predict_step(self, batch, batch_idx, output_scores: bool = False):
        ids, text_tokens = batch
        text_tokens = {k: v.to(self.device) for k, v in text_tokens.items()}

        outputs = self.clm.generate(
            **text_tokens,
            penalty_alpha=0.6,
            top_k=4, max_new_tokens=128,
            return_dict_in_generate=True,
            output_scores=output_scores
        )

        return ids, (outputs.sequences, outputs.scores)

    # Adapted from https://github.com/huggingface/transformers/blob/04ab5605fbb4ef207b10bf2772d88c53fc242e83/src/transformers/models/gpt2/modeling_gpt2.py#L1448
    def _get_sequence_length(self, sequence):
        pad_token_id = self.clm.config.pad_token_id

        return (
            torch.ne(
                sequence,
                pad_token_id
            ).sum(-1) - 1
        ).to(self.device)

    # Refer this comment:
    # https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/14?
    @torch.no_grad()
    def get_action_and_values(self, state_tokens):
        """NOTE: Ensure that the states are left padded.
        """
        pad_token_id = self.clm.config.pad_token_id
        _, (action_ids, action_scores) = self.predict_step(
            (None, state_tokens),
            None, output_scores=True
        )

        # Get only the generated token_ids
        input_len = state_tokens.input_ids.size(1)
        action_ids = action_ids[:, input_len:]

        # Calc the log_probs
        per_token_log_probs = self.clm.compute_transition_scores(
            action_ids, action_scores, normalize_logits=True
        )

        non_padding_mask =  torch.ne(action_ids, pad_token_id)
        action_length = torch.ne(action_ids, pad_token_id).sum(-1)
        unnormalized_log_probs = (per_token_log_probs*non_padding_mask).sum(dim=-1)

        log_probs = unnormalized_log_probs / action_length

        # Calc value
        state_tokens = {k: v.to(self.device) for k, v in state_tokens.items()}
        # Since we pad left size, the last state token is always at the end
        raw_values = self.clm.transformer(
            **state_tokens).last_hidden_state[..., -1, :].contiguous()
        
        values = self.value_head(raw_values).squeeze()

        return action_ids, log_probs, values

    def get_log_probs(self, state_tokens, action_tokens):
        """NOTE: Ensure
            - the states are left padded and actions
            are right padded.
            - action_tokens have labels
        """
        # Set the state labels to -100
        state_tokens['labels'] = torch.full_like(state_tokens.input_ids, -100)
        # Build action mask
        mask = action_tokens.attention_mask.logical_not()
        action_tokens.labels[mask] = -100
        
        action_lengths = action_tokens.attention_mask.sum(-1).to(self.device)
        # Combine the state and action
        for key in action_tokens.keys():
            action_tokens[key] = torch.cat(
                (state_tokens[key], action_tokens[key]),
                dim=-1
            ).to(self.device)

        # Get the action logits
        output = self(action_tokens)
        logits = output.logits

        # Calculations based on:
        # https://github.com/huggingface/transformers/blob/04ab5605fbb4ef207b10bf2772d88c53fc242e83/src/transformers/models/gpt2/modeling_gpt2.py#L1100
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = action_tokens['labels'][..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        log_probs = -loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(shift_labels.shape).sum(-1) / action_lengths

        # Check to see if our calcs are valid
        assert -log_probs.mean() == output.loss, "Check log_probs calculation"

        return log_probs

    def get_values(self, state_tokens):
        state_tokens = {k: v.to(self.device) for k, v in state_tokens.items()}

        # Since we pad left size, the last state token is always at the end
        raw_values = self.clm.transformer(
            **state_tokens).last_hidden_state[..., -1, :].contiguous()
        
        values = self.value_head(raw_values).squeeze()

        return values

    def configure_optimizers(self):
        param_dicts = [
            {"params": self.clm.parameters()},
            {
                "params": self.value_head.parameters(),
                # Use a higher LR for the value head as it has to be
                # learnt from scratch
                "lr": self.lr * 1e2
            }
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

        labels = text_tokens['labels'].to(self.device)

        # Weighted loss to increase recall
        num_pos = (labels > 0).sum(dim=0)
        num_neg = labels.size(0) - num_pos
        pos_weight = num_neg / (num_pos+1)
        loss = F.binary_cross_entropy_with_logits(
            output.logits,
            labels,
            pos_weight=pos_weight
        )

        # Compute the metrics
        granular_metric = MultilabelF1Score(num_labels=self.num_labels, average=None).to(self.device)
        overall_metric = MultilabelF1Score(num_labels=self.num_labels, average='micro').to(self.device)
        macro_metric = MultilabelF1Score(num_labels=self.num_labels).to(self.device)
        reshaped_targets = labels
        pred_labels = output.logits.sigmoid()

        return {
            'loss': loss,
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
