import numpy as np

import torch
from torch.optim import AdamW
from torch.nn import functional as F
import torch.nn
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassF1Score

from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers import pipelines
from tqdm import tqdm

from src.dataloader import dataset_generator
import os
import omegaconf
from glob import glob
import wandb
import pickle

    
def save_best_model(model, pre_trained_model):
    print("Saving final model...")
    model.module.save_pretrained("./model_save/%s.pt"%(pre_trained_model))


def model_train(config, config_name):
    
    torch.manual_seed(config.seed)
    wandb.init(project = "FinSimESG_Final", name = config_name)
    wandb.config.update(config)
    
    DataGenerator = dataset_generator(train_data_dir = config.datamodule.train_path,
                                      val_data_dir = config.datamodule.val_path,
                                      tokenizer_type = config.pretrained_model,
                                      batch_size = config.datamodule.batch_size,
                                      label_dict = label2id)
    train_dataset, val_dataset = DataGenerator.dataset_return()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(config.pretrained_model, num_labels = DataGenerator.label_num, ignore_mismatched_sizes=True)
    model = torch.nn.DataParallel(model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate)
    #epochs = config.train.num_epochs
    epochs = 18
    acc_metric = Accuracy(task="multiclass", num_classes=DataGenerator.label_num, top_k=1).to(device)
    weighted_f1_metric =  MulticlassF1Score(num_classes=DataGenerator.label_num, average="weighted").to(device)
    macro_f1_metric = MulticlassF1Score(num_classes=DataGenerator.label_num, average="macro").to(device)

    min_val_loss = 1000
    min_acc = -1
    min_f1 = -1

    for epoch in range(epochs):
        model.train()
        train_loss = []
        
        for batch in tqdm(train_dataset):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(loss.item())
            wandb.log({'train loss': loss.item()})
        print("=======> Epoch: %d, Average loss: %.6f"%(epoch, np.mean(train_loss)))

        model.eval()
        test_loss = []
        acc_ = []
        f1_ = []
        wf1_ = []
        for batch in tqdm(val_dataset):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss.mean()
            test_loss.append(loss.item())
            acc = acc_metric(outputs.logits, batch['labels'])
            f1 = macro_f1_metric(outputs.logits, batch['labels'])
            wf1 = weighted_f1_metric(outputs.logits, batch['labels'])
            wandb.log({'valid loss' : loss.item(), 'valid acc' : acc, 'valid macro f1' : f1, 'valid weighted f1' : wf1})
            acc_.append(acc)
            f1_.append(f1)
            wf1_.append(wf1)
        print("=======> Average test loss: %.6f"%(np.mean(test_loss)))

        if np.mean(test_loss) < min_val_loss:
            best_model = model
            best_model_acc = torch.mean(torch.stack(acc_))
            best_model_f1 = torch.mean(torch.stack(f1_))
            best_model_wf1 = torch.mean(torch.stack(wf1_))
            wandb.log({'best model acc': best_model_acc, 'best_model_macro_f1' : best_model_f1, 'best_model_weighted_f1' : best_model_wf1})
    save_best_model(best_model,config_name)
    wandb.finish()
    del model



if __name__ == "__main__":
    import warnings
    warnings.filterwarnings(action="ignore")

    with open('/workspace/mlesg/trainer_v02/label_dict', 'rb') as fr:
        label2id = pickle.load(fr)

    
    min_val_loss = 1000
    config_dir = "src/config"
    config_paths = glob(os.path.join(config_dir, "*.yaml"))
    config_paths = config_paths
    os.environ["WANDB_API_KEY"]="c868a448cc5ddc72551513a6703020aef5044dd6"
    wandb.login()


    for config_path in tqdm(config_paths):
        config = omegaconf.OmegaConf.load(config_path)
        config_name = config_path.split("/")[2].replace(".yaml", "")
        check_model = './model_save/'+config_name+'.pt'
        if os.path.exists(check_model):
            pass
        else:
            model_train(config, config_name)
            torch.cuda.empty_cache()

        