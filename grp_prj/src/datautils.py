import os
import torch
import pytorch_lightning as pl

from copy import deepcopy
from transformers import AutoTokenizer, BatchEncoding, pipeline
from torch.utils.data import DataLoader, Dataset
from convokit import Corpus, Conversation, Utterance, download
from sklearn.model_selection import train_test_split
from typing import Dict, List, Literal, Optional, Sequence, Tuple



class DialogDataset(Dataset):
    def __init__(
        self,
        convos: List[Conversation]
    ):
        self.convos = convos

    def __len__(self):
        return len(self.convos)

    def __getitem__(self, index):
        return self.convos[index]


class DialogBatcher:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        purpose_text: str = "",
        needs_targets: bool = True,
        is_pursuader: bool = True
    ):
        """
        Args:
            tokenizer (AutoTokenizer): The `Transformers` tokenizer
            to be used.
            purpose_text (str): Add some purpose information to the head of every
            conversation. Defaults to "".
            needs_targets (bool): Do labels have to be generated?
            Defaults to True.
            is_pursuader (bool): Are you generating data to train the persuader?
            Defaults to True.
        
        Note:
        Ensure the first value in tokenizer's special_tokens_map['additional_special_tokens']
        is the persuader and the second the persuadee. Also ensure there are only 2 values.
        """
        self.tokenizer: AutoTokenizer = tokenizer
        self.needs_targets = needs_targets
        self.is_pursuader = is_pursuader
        self.purpose_text = purpose_text

        # GPT Models don't have a padding requirement, hence this is not set
        # GPT Models have all special tokens set to eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_samples(
        self,
        convo: Conversation
    ) -> Tuple[List[int], List[BatchEncoding]]:
        """Generates samples for the conversation by concatenating history to each
        utterance. Further we use the `self.is_persuader` flag to generate appropriate
        persuader or persuadee finetuning data.
        Note: No need to pad fo causal LMs.
        https://github.com/huggingface/transformers/issues/2630#issuecomment-684512764
        """
        role_markers = self.tokenizer.special_tokens_map['additional_special_tokens']
        base_utt = f"{self.tokenizer.bos_token} {self.purpose_text}\n\n"
        tokenized_hist = self.tokenizer(base_utt)
        tokenized_hist['labels'] = [-100] * len(tokenized_hist['input_ids'])

        ids: List[int] = list()
        batch_encodings: List[BatchEncoding] = list()
        for utt in convo.iter_utterances():
            utt_text = f"{role_markers[utt.meta['role']]} {utt.text}"

            # If the utterance is for the role in question 
            if (self.is_pursuader and utt.meta['role']==0) or \
                (not self.is_pursuader and utt.meta['role']==1):

                formatted_utterance = f"{utt_text}{self.tokenizer.eos_token}"
                tokenized_utt = self.tokenizer(
                    formatted_utterance,
                    text_target=formatted_utterance
                )
                
            else:
                tokenized_utt = self.tokenizer(utt_text)
                tokenized_utt['labels'] = [-100] * len(tokenized_utt['input_ids'])

            # Concatenate hist to the current
            for key in tokenized_utt.keys():
                tokenized_utt[key] = deepcopy(tokenized_hist[key]+tokenized_utt[key])

            # Update the base and data returning
            if (self.is_pursuader and utt.meta['role']==0) or \
                (not self.is_pursuader and utt.meta['role']==1):

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

        return ids, batch_encodings

    def __call__(self, batch: Sequence[Conversation]):
        """Use this function as the `collate_fn` for the torch Dataloader.
        """
        ids: List[int] = list()
        batch_encodings: List[BatchEncoding] = list()
        for convo in batch:
            # Get the encoded utterances
            utt_ids, utt_batch_encodings = self._build_samples(convo)

            ids.extend(utt_ids)
            batch_encodings.extend(utt_batch_encodings)

        # Tensorify, pad and aggregate
        batch_encodings = self.tokenizer.pad(
            batch_encodings,
            padding=True
        )

        # Pad the labels
        batch_max_length = len(batch_encodings['input_ids'][0])

        for idx, target in enumerate(batch_encodings['labels']):
            batch_encodings['labels'][idx] = \
                target + [-100] * (batch_max_length-len(target))

        # Tensorify, pad and aggregate
        batch_encodings = self.tokenizer.pad(
            batch_encodings,
            padding=True,
            return_tensors='pt'
        )

        return ids, batch_encodings

class DialogDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data: str, /,
        batcher: DialogBatcher,
        batch_size: int = 16,
        split_data: bool = True
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
        """
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
