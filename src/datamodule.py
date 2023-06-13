import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from collections import Counter
from typing import List, Tuple, Dict

from .label_dict import LABEL_IDX_DICT, IDX_LABEL_DICT


class ESGDataModule:
    def __init__(
        self,
        train_path: str,
        val_path: str = None,
        test_path: str = None,
        tokenizer_name: str = None,
        use_cls_weight: bool = True,
        batch_size: int = 32,
    ):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.use_cls_weight = use_cls_weight
        self.loss_weights = None
        self.batch_size = batch_size
        self.tokenizer_name = tokenizer_name
        self.label_idx_dict = LABEL_IDX_DICT
        self.idx_label_dict = IDX_LABEL_DICT

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.setup()

    def setup(self):
        # train set
        train_dict = self._load_data(self.train_path)
        train_encodings = self.tokenizer(
            train_dict["title"],
            add_special_tokens=True,
            text_pair=train_dict["content"],
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        self.trainset = ESGDataset(train_encodings, train_dict["label"], self.label_idx_dict)

        if self.use_cls_weight:
            target_cnt_dict = Counter([self.label_idx_dict[label] for label in train_dict["label"]])
            target_cnt = [target_cnt_dict[idx] for idx in self.idx_label_dict]

            loss_weights = [1 - (x / sum(target_cnt)) for x in target_cnt]
            self.loss_weights = torch.FloatTensor(loss_weights)

        # valid set
        val_dict = self._load_data(self.val_path)
        val_encodings = self.tokenizer(
            val_dict["title"],
            add_special_tokens=True,
            text_pair=val_dict["content"],
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        self.valset = ESGDataset(val_encodings, val_dict["label"], self.label_idx_dict)

        # test set
        self.testset = None
        if self.test_path:
            test_dict = self._load_data(self.test_path)
            test_encodings = self.tokenizer(
                test_dict["title"],
                add_special_tokens=True,
                text_pair=test_dict["content"],
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            self.testset = ESGDataset(test_encodings, test_dict["label"], self.label_idx_dict)

    def train_dataloader(self):
        return DataLoader(self.trainset, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valset, shuffle=True, batch_size=self.batch_size)

    def test_dataloader(self):
        if self.testset:
            return DataLoader(self.testset, shuffle=False, batch_size=self.batch_size)
        return None

    def get_num_labels(self) -> int:
        return len(self.label_idx_dict)

    def _load_data(self, data_path: str) -> Dict:
        if "json" in data_path:
            df = pd.read_json(data_path)
        elif "csv" in data_path:
            df = pd.read_csv(data_path)
        elif "parquet" in data_path:
            df = pd.read_parquet(data_path, engine="pyarrow")

        titles = df["news_title"].tolist()
        contents = df["news_content"].tolist()
        labels = df["ESG_label"].tolist()

        return {"title": titles, "content": contents, "label": labels}


class ESGDataset(Dataset):
    def __init__(self, encodings: List[int], labels: List[str], label_idx_dict: Dict[str, int]):
        self.encodings = encodings
        self.labels = labels
        self.label_idx_dict = label_idx_dict

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        label = self.labels[idx]
        item["labels"] = torch.tensor(self.label_idx_dict[label])
        return item
