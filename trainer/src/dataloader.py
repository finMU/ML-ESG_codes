import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import pickle


class data_agg(torch.utils.data.Dataset):
    def __init__(self,
                 text,
                 labels):
        
        self.text = text
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.text.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

class dataset_generator():
    def __init__(self,
                 train_data_dir,
                 val_data_dir,
                 tokenizer_type,
                 batch_size,
                 label_dict):
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.tokenizer_type = tokenizer_type
        self.batch_size = batch_size
        self.label_dict = label_dict
        self.label_num = len(label_dict)

    def dataset_return(self):
        
        train_dataloader = self.datasets(self.train_data_dir)
        val_dataloader = self.datasets(self.val_data_dir)
        return train_dataloader, val_dataloader
        
    def datasets(self, data_path):
        data = pd.read_parquet(data_path)
        news_title, news_texts, labels = self.pre_process(data)
        encodings= self.tokenize(news_title, news_texts)
        dataset = data_agg(encodings, labels)
        dataloader = DataLoader(dataset, shuffle = True, batch_size = self.batch_size)            
        return dataloader
    
        
    def pre_process(self, data):
        data['ESG_label'] = data['ESG_label'].map(self.label_dict)            
        return list(data['news_title']), list(data['news_content']), list(data['ESG_label'])


    def tokenize(self, news_title, news_text):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_type)
        train_encodings = tokenizer(news_title,add_special_tokens=True, text_pair=news_text, truncation = True, padding = True, return_tensors = "pt")
        return train_encodings