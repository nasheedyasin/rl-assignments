import torch
import pytorch_lightning as pl

from typing import Dict
from torch.nn import functional as F
from torchmetrics.text.rouge import ROUGEScore
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class PersuasionRewards(pl.LightningModule):
    def __init__(
        self,
        a1, a2, a3, a4, /, *, 
        tokenizer: AutoTokenizer,
        emotion_cls: AutoModelForSequenceClassification,
        persuasion_cls: AutoModelForSequenceClassification,
        rouge_mode: str = 'rougeL'
    ):
        """
        Args:
            a1, a2, a3, a4 (int): The weights for the various aspects of the rewards.
            (repetitiveness, consistency, emotion, persuasion)
        """
        super().__init__()

        self.tokenizer = tokenizer
        self.persuasion_cls = persuasion_cls
        self.emotion_cls = emotion_cls
        self.rouge_metric = ROUGEScore(use_stemmer=True, rouge_keys=rouge_mode)
        
        self.weight_tensor = torch.tensor([a1, a2, a3, a4])

    def forward(self, text_tokens):
        # Push all inputs to the device in use
        text_tokens = {k: v.to(self.device) for k, v in text_tokens.items()}

        emo_logits = self.emotion_cls(**text_tokens).logits
        per_logits = self.persuasion_cls(**text_tokens).logits

        return emo_logits, per_logits

    def predict_step(self, batch):
        # Right now only supports 1 sample at a time
        prev_gen, curr_gen, gold_response = batch

        # Calculate repetitiveness reward
        rep_reward = self.rouge_metric(curr_gen, prev_gen)['rougeL_fmeasure']
        rep_reward = rep_reward

        # Calculate consistency reward
        const_reward = self.rouge_metric(curr_gen, gold_response)['rougeL_fmeasure']
        const_reward = const_reward

        # Prep data for emotion and persuasion reward calc
        text_tokens = self.tokenizer(
            [curr_gen, gold_response],
            truncation=True,
            padding=True,
            return_tensors='pt'
        )

        with torch.no_grad():
            emo_logits, per_logits = self(text_tokens)

        curr_gen_emo_logits, gold_emo_logits = emo_logits
        curr_gen_per_logits, gold_per_logits = per_logits

        # Calculate emotion reward (range: 0-1)
        # Calculated as the ratio of the generated response belonging to the gold
        # repsonse class and the gold response beloning to the gold repsonse class. 
        curr_gen_log_probs = curr_gen_emo_logits.log_softmax(dim=-1)
        gold_log_probs = gold_emo_logits.log_softmax(dim=-1)

        gold_max, gold_argmax = gold_log_probs.max(dim=-1)
        curr_gen_max = curr_gen_log_probs.gather(-1, gold_argmax.unsqueeze(-1)).squeeze()

        emo_reward = torch.clip((curr_gen_max - gold_max).exp(),max=1).cpu()

        # Calculate persuasion reward (range: 0-1)
        # Calculated as the cosine similarity bewteen the generated reponses's label tensor
        # and the gold response's label tensor
        per_reward = F.cosine_similarity(
            curr_gen_per_logits.sigmoid(),
            gold_per_logits.sigmoid(),
            dim=-1
        ).cpu()

        unweighted_reward_tensor = torch.stack(
            [-rep_reward, const_reward, emo_reward, per_reward], dim=-1
        )

        reward = (unweighted_reward_tensor*self.weight_tensor).sum(dim=-1)

        return reward
