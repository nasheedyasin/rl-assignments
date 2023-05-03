import os
import torch
import random
import pytorch_lightning as pl

from tqdm import tqdm
from copy import deepcopy
from transformers import AutoTokenizer, BatchEncoding, pipeline
from torch.utils.data import DataLoader, Dataset
from convokit import Corpus, Conversation, download
from sklearn.model_selection import train_test_split
from typing import List, Literal, Optional, Sequence, Tuple


random.seed(10)

ROLES = ('persuader', 'persuadee', 'both')

class DialogDataset(Dataset):
    def __init__(
        self,
        convos: List[Conversation],
        tokenizer: AutoTokenizer,
        purpose_text: str = "",
        role:  Literal['both', 'persuader', 'persuadee'] = 'both',
        shuffle: bool = False,
        max_length=768
    ):        
        """
        Args:
            tokenizer (AutoTokenizer): The `Transformers` tokenizer
            to be used.
            purpose_text (str): Add some purpose information to the head of every
            conversation. Defaults to "".
            role (bool): Are you generating data to train the persuader or persuadee?
            Defaults to 'persuader'.
        
        Note:
        Ensure the first value in tokenizer's special_tokens_map['additional_special_tokens']
        is the persuader and the second the persuadee. Also ensure there are only 2 values.
        """
        self.tokenizer: AutoTokenizer = tokenizer
        self.role = role
        self.purpose_text = purpose_text
        self.max_length = max_length

        # GPT Models don't have a padding requirement, hence this is not set
        # GPT Models have all special tokens set to eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.ids: List[str] = list()
        self.utterances: List[BatchEncoding] = list()
        # Get the utterances
        for convo in tqdm(convos, desc='Enumerating the Utterances'):
            ids, utts = self._build_samples(convo)

            self.ids.extend(ids)
            self.utterances.extend(utts)

        # Shuffle the utterances
        if shuffle:
            _temp = list(zip(self.ids, self.utterances))
            random.shuffle(_temp)

            self.ids, self.utterances = zip(*_temp)

    def __len__(self):
        return len(self.utterances)

    def _build_samples(
        self,
        convo: Conversation,
    ) -> Tuple[List[str], List[BatchEncoding]]:
        """Generates samples for the conversation by concatenating history to each
        utterance. Further we use the `self.is_persuader` flag to generate appropriate
        persuader or persuadee finetuning data.
        Note: 
        """
        role_markers = self.tokenizer.special_tokens_map['additional_special_tokens']

        base_utt = f"{self.tokenizer.bos_token} {self.purpose_text} {role_markers[0]}"
        tokenized_hist = self.tokenizer(base_utt)
        tokenized_hist['labels'] = [-100] * len(tokenized_hist['input_ids'])

        ids: List[str] = list()
        batch_encodings: List[BatchEncoding] = list()
        for utt in convo.iter_utterances():
            # If the utterance is for the role in question 
            if self.role == 'both' or (self.role=='persuader' and utt.meta['role']==0) or \
                (self.role=='persuadee' and utt.meta['role']==1):

                formatted_utterance = f"{utt.text}{self.tokenizer.eos_token}"
                tokenized_utt = self.tokenizer(
                    formatted_utterance,
                    text_target=formatted_utterance
                )
                
            else:
                tokenized_utt = self.tokenizer(utt.text)
                tokenized_utt['labels'] = [-100] * len(tokenized_utt['input_ids'])

            # Concatenate hist to the current
            for key in tokenized_utt.keys():
                tokenized_utt[key] = deepcopy(tokenized_hist[key]+tokenized_utt[key])
                # Truncate from the left
                if len(tokenized_utt[key]) > self.max_length:
                    truncation_length = len(tokenized_utt[key]) - self.max_length
                    tokenized_utt[key] = tokenized_utt[key][truncation_length: ]

            # Update the base and data returning
            if self.role == 'both' or (self.role=='persuader' and utt.meta['role']==0) or \
                (self.role=='persuadee' and utt.meta['role']==1):

                # Append the utterance    
                ids.append(utt.id)
                batch_encodings.append(tokenized_utt)

                # Remove the EOS token from history 
                for key in tokenized_utt.keys():
                    tokenized_hist[key] = deepcopy(tokenized_utt[key][:-1])

                # Update the labels
                tokenized_hist['labels'] = [-100] * len(tokenized_hist['input_ids'])

            else:
                tokenized_hist = deepcopy(tokenized_utt)

            # Add the next role Label (only 2 roles are present)
            tokenized_hist['input_ids'].append(
                self.tokenizer.convert_tokens_to_ids(
                    role_markers[int(not utt.meta['role'])]
                )
            )
            tokenized_hist['attention_mask'].append(1)
            tokenized_hist['labels'].append(-100)

        return ids, batch_encodings

    def __getitem__(self, index) -> Tuple[int, BatchEncoding]:
        return self.ids[index], self.utterances[index]


class DialogBatcher:
    def __init__(
        self,
        tokenizer: AutoTokenizer
    ):
        """
        Args:
            tokenizer (AutoTokenizer): The `Transformers` tokenizer
            to be used.
        
        Note:
        Ensure the first value in tokenizer's special_tokens_map['additional_special_tokens']
        is the persuader and the second the persuadee. Also ensure there are only 2 values.
        """
        self.tokenizer: AutoTokenizer = tokenizer

        # GPT Models don't have a padding requirement, hence this is not set
        # GPT Models have all special tokens set to eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


    def __call__(self, batch: Sequence[Tuple[str, BatchEncoding]]):
        """Use this function as the `collate_fn` for the torch Dataloader.
        """
        ids = [i[0] for i in batch]
        batch_encodings: List[BatchEncoding] = [i[1] for i in batch]

        # Pad and aggregate
        batch_encodings = self.tokenizer.pad(
            batch_encodings,
            padding='max_length',
            max_length=768
        )

        # Pad the labels
        batch_max_length = len(batch_encodings['input_ids'][0])

        for idx, target in enumerate(batch_encodings['labels']):
            batch_encodings['labels'][idx] = \
                target + [-100] * (batch_max_length-len(target))

        # Tensorify
        batch_encodings.convert_to_tensors('pt')

        return ids, batch_encodings

class DialogDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data: str, /,
        batcher: DialogBatcher,
        batch_size: int = 16,
        split_data: bool = True, *,
        purpose_text: str = ""
    ):
        """
        Args:
            data (str): path to or name of the dataset.
            batcher (SynBatcher): Custom data batching logic.
            split_data (bool): Whether or not to split the data into train and test
            sets.
                Eval mode: Only the test set will be used.
                Train mode: Only the train set will be used. This train set will be further
                divided into train and validation sets.
            purpose_text (str): Add some purpose information to the head of every
            conversation. Defaults to "".
        """
        super().__init__()
        self.purpose_text = purpose_text

        if os.path.exists(data):
            self.data_path = data
        else:
            self.data_path = download(data)

        self.batcher = batcher
        self.batch_size = batch_size
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
            self.train_dataset = DialogDataset(
                cnv_train,
                self.batcher.tokenizer,
                purpose_text = self.purpose_text,
                shuffle=True
            )
            # Validation Dataset
            self.val_dataset = DialogDataset(
                cnv_val,
                self.batcher.tokenizer,
                purpose_text = self.purpose_text
            )

        if stage == "test" or stage is None:
            if self.split_data:
                _, convs = train_test_split(
                    convs,
                    train_size=train_fraction,
                    random_state=44
                )

            self.test_dataset = DialogDataset(
                convs,
                self.batcher.tokenizer,
                purpose_text = self.purpose_text
            )

        if stage == "predict" or stage is None:
            self.pred_dataset = DialogDataset(
                convs,
                self.batcher.tokenizer,
                purpose_text = self.purpose_text
            )

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
