import os

import torch
from dataclasses import dataclass
from datasets import load_from_disk
import lightning as L
from torch.utils.data import DataLoader

from .preprocess import PREPROCESS
from .load import LOAD

KEYS = [
    "enc_input_ids",
    "enc_attention_mask",
    "dec0_tgt",
    "dec_attention_mask",
    "dec1_tgt",
    "enc_tgt",
    "dec_input_ids",
]


@dataclass
class Collator:
    def __call__(self, features):
        # features: list of dict
        batch = {}
        for key in features[0].keys():
            if key in KEYS:
                batch[key] = torch.tensor([feature[key] for feature in features])
            else:
                batch[key] = [feature[key] for feature in features]

        return batch


class DataModule(L.LightningDataModule):
    def __init__(self, args, tokenizers) -> None:
        super().__init__()

        self.args = args
        self.tokenizers = tokenizers

    def setup(self, stage) -> None:
        dir = os.path.join(
            os.getcwd(), "data", self.args.data, self.args.architecture, "cache"
        )
        # remove_columns = ["src", "morph_tgt", "tag_tgt"]
        remove_columns = []

        def load_and_preprocess_data(split_name, remove_columns, tokenizers):
            if os.path.exists(os.path.join(dir, split_name)):
                dataset = load_from_disk(os.path.join(dir, split_name))
            else:
                os.makedirs(os.path.join(dir, split_name), exist_ok=True)
                dataset = LOAD[self.args.data](split_name)

                dataset = dataset.map(
                    PREPROCESS[self.args.data],
                    remove_columns=remove_columns,
                    fn_kwargs={
                        "tokenizers": tokenizers,
                    },
                    load_from_cache_file=True,
                    keep_in_memory=True,
                    desc="Pre-processing",
                    batched=True,
                )

                dataset.save_to_disk(os.path.join(dir, split_name))

            return dataset

        if stage == "fit":
            self.train = load_and_preprocess_data(
                "train", remove_columns, self.tokenizers
            )
            self.valid = load_and_preprocess_data(
                "valid", remove_columns, self.tokenizers
            )

        if stage == "test":
            self.test = load_and_preprocess_data(
                "test", remove_columns, self.tokenizers
            )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            collate_fn=Collator(),
            batch_size=self.args.trainer_args["limit_train_batches"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            collate_fn=Collator(),
            batch_size=self.args.trainer_args["limit_val_batches"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            collate_fn=Collator(),
            batch_size=self.args.trainer_args["limit_test_batches"],
        )

    def predict_dataloader(self):
        # what's the difference between this and test?
        return DataLoader(
            self.test,
            collate_fn=Collator(),
            batch_size=self.args.trainer_args["limit_predict_batches"],
        )
