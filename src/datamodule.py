import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from collections import Counter
from typing import List, Tuple, Dict
from transformers import AutoTokenizer


class ESGDataModule:
    def __init__(
        self,
        train_path: str,
        val_path: str = None,
        test_path: str = None,
        tokenizer_name: str = None,
        use_title: bool = True,
        use_cls_weight: bool = True,
        batch_size: int = 32,
    ):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.use_title = use_title
        self.use_cls_weight = use_cls_weight
        self.loss_weights = None
        self.batch_size = batch_size
        self.tokenizer_name = tokenizer_name
        self.label_idx_dict = None

        self.setup()

    def setup(self):
        # train set
        # TODO: label_dict 불러와서 고정시켜 놓을것
        train_texts, train_labels, label_idx_dict = self._load_data(self.train_path, self.use_title)
        self.label_idx_dict = label_idx_dict
        self.idx_label_dict = {idx: label for label, idx in self.label_idx_dict.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        train_encodings = self.tokenizer(
            train_texts,
            add_special_tokens=True,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        self.trainset = ESGDataset(train_encodings, train_labels, self.label_idx_dict)
        # valid set
        valid_texts, valid_labels, vlabel_idx_dict = self._load_data(self.val_path, self.use_title)

        valid_encodings = self.tokenizer(
            valid_texts,
            add_special_tokens=True,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        self.validset = ESGDataset(valid_encodings, valid_labels, self.vlabel_idx_dict)

        if self.use_cls_weight:
            target_cnt_dict = Counter([self.label_idx_dict[label] for label in train_labels])
            target_cnt = [target_cnt_dict[idx] for idx in self.idx_label_dict]

            loss_weights = [1 - (x / sum(target_cnt)) for x in target_cnt]
            self.loss_weights = torch.FloatTensor(loss_weights)

    def train_dataloader(self):
        return DataLoader(self.trainset, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validset, shuffle=True, batch_size=self.batch_size)

    def test_dataloader(self):
        pass

    def get_num_labels(self):
        return len(self.label_idx_dict)

    def _load_data(self, data_path: str, use_title: bool = True) -> Tuple[List[str], List[str]]:
        if "json" in data_path:
            df = pd.read_json(data_path)
        elif "csv" in data_path:
            df = pd.read_csv(data_path)
        elif "parquet" in data_path:
            df = pd.read_parquet(data_path, engine="pyarrow")

        texts = df["news_content"].tolist()
        labels = df["ESG_label"].tolist()

        # TODO: title, text 따로 리턴하게 만들것
        if use_title:
            titles = df["news_title"].tolist()

            assert len(texts) == len(titles)
            texts = [f"{title} [SEP] {text}" for title, text in zip(titles, texts)]

        label_idx_dict = {label: idx for idx, label in enumerate(set(labels))}
        return texts, labels, label_idx_dict


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
