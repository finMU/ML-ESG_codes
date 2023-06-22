import os
import glob
import omegaconf

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification

from datamodule import ESGDataModule
from trainer import Trainer
from utils import fix_seed

import warnings

warnings.filterwarnings(action="ignore")


def main(config):
    fix_seed(config.seed)
    scaler = torch.cuda.amp.GradScaler() if config.amp else None

    # datamodule
    dm_params = {
        "train_path": config.datamodule.train_path,
        "val_path": config.datamodule.val_path,
        "tokenizer_name": config.pretrained_model,
        "use_cls_weight": config.datamodule.use_cls_weight,
        "batch_size": config.datamodule.batch_size,
    }

    dm = ESGDataModule(**dm_params)

    # model
    model = AutoModelForSequenceClassification.from_pretrained(
        config.pretrained_model, num_labels=dm.get_num_labels(), ignore_mismatched_sizes=True
    )

    # trainer & training
    model_name = config.pretrained_model.split("/")[-1]
    config.model.model_name = model_name
    trainer = Trainer(config, model, dm, scaler)
    version = trainer.fit()
    print(version)


if __name__ == "__main__":
    config_path = "./config/train_config.yaml"
    config = omegaconf.OmegaConf.load(config_path)

    main(config)
