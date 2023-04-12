import os
import json
import torch
import string
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from tqdm import tqdm
from abc import ABC, abstractmethod
from convokit import Corpus, download
from transformers import AutoTokenizer, pipeline
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from typing import Dict, List, Literal, Optional, Sequence, Tuple



class DialogDataset(Dataset):
    pass

class DialogBatcher:
    def __init__(
        self,
        tnkzr_path: str,
        has_targets: bool = True,
        concat_title_and_generation: bool = False
    ):
        """
        Args:
            tnkzr_path (str): Path to load the `Transformers` tokenizer
            to be used.
            has_targets (bool): Does the dataset have target information.
            Defaults to True.
            concat_title_and_generation (bool): Whether to concatenate the title
            to the generation. Could be useful for non-autoregressive models.
        """
        self.has_targets = has_targets
        self.tokenizer = AutoTokenizer.from_pretrained(tnkzr_path)

        # GPT Models don't have a padding requirement, hence this is not set
        # GPT Models have all special tokens set to eos_token
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left'

        self.concat_title_and_generation = concat_title_and_generation

    def __call__(self, batch: Sequence):
        """Use this function as the `collate_fn` mentioned earlier.
        """
        ids = torch.tensor([int(sample[0]) for sample in batch], dtype=torch.int32)

        if self.concat_title_and_generation:
            text_tokens = self.tokenizer(
                [f"{sample[1]} {sample[2]}" for sample in batch],
                padding="max_length",
                truncation=True,
                return_tensors='pt'
            )
        else:
            # The tokenization is done with the title as sentence 1 and
            # generation as sentence 2
            text_tokens = self.tokenizer(
                [sample[1] for sample in batch], # The title
                [sample[2] for sample in batch], # The generation
                padding="max_length",
                truncation=True,
                return_tensors='pt'
            )

        if not self.has_targets:
            return ids, text_tokens

        targets = torch.tensor(
            [sample[3]for sample in batch],
            dtype=torch.long
        )

        return ids, text_tokens, targets

# CFT = Causal Finetune
class DialogDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data: str, /,
        batcher: DialogBatcher,
        batch_size: int = 16,
        is_pursuader: bool = True,
        split_data: bool = True
    ):
        """
        Args:
            data (str): path to or name of the dataset.
            batcher (SynBatcher): Custom data batching logic.
            is_pursuader (bool): Configure datamodule for pursuader. Defaults to True.
            split_data (bool): Whether or not to split the data into train and test
            sets.
                Eval mode: Only the test set will be used.
                Train mode: Only the train set will be used. This train set will be further
                divided into train and validation sets.
        """
        if os.path.exists(data):
            self.data_path = data
        else:
            self.data_path = download(data)

        self.batcher = batcher
        self.batch_size = batch_size
        self.is_pursuader = is_pursuader
        self.split_data = split_data

    def setup(
        self,
        stage: Optional[str] = None,
        train_fraction: float = 0.75
    ) -> None:
        """Read in the data csv file and perform splitting here.
        Args:
            train_fraction (float): Fraction of the conversations to use
            as training data.
        """
        corpus = Corpus(filename=self.data_path)
        convs = [conv for conv in corpus.iter_conversations()]

        if stage == "fit" or stage is None:
            cnv_train, cnv_val = train_test_split(
                convs,
                train_size=train_fraction,
                random_state=44
            )

            # Split the train set again into train and validation if test
            # dataset creation was required.
            if self.split_data:
                cnv_train, cnv_val = train_test_split(
                    cnv_train,
                    train_size=train_fraction,
                    random_state=44
                )

            # Train Dataset
            self.train_dataset = DialogDataset(cnv_train)
            # Validation Dataset
            self.val_dataset = DialogDataset(cnv_val)

        if stage == "test" or stage is None:
            if self.split_data:
                _, convs = train_test_split(
                    convs,
                    train_size=train_fraction,
                    random_state=44
                )

            self.test_dataset = DialogDataset(convs)

        if stage == "predict" or stage is None:
            self.pred_dataset = DialogDataset(convs)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.batcher,
            shuffle=True
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.batcher
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.batcher
        )

    def predict_dataloader(self):
        return DataLoader(
            self.pred_dataset,
            batch_size=self.batch_size,
            collate_fn=self.batcher
        )
