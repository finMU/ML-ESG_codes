import os
import glob
import random
import logging
from typing import *

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassF1Score
from tqdm import tqdm

from utils import AverageMeter


class Trainer:
    def __init__(self, config, model, dm, scaler):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # model
        self.model_name = config.model.model_name
        self.model = model.to(self.device)
        self.model = nn.DataParallel(self.model)

        # datamodule(dm)
        self.dm = dm
        self.train_loader = self.dm.train_dataloader()
        self.val_loader = self.dm.val_dataloader()

        self.scaler = scaler

        # optimizer
        self.optimizer = self.configure_optimizers(lr=self.config.train.learning_rate)

        # metric
        self.acc_metric = Accuracy(
            task="multiclass", num_classes=self.dm.get_num_labels(), top_k=1
        ).to(self.device)
        self.weighted_f1_metric = MulticlassF1Score(
            num_classes=self.dm.get_num_labels(), average="weighted"
        ).to(self.device)
        self.macro_f1_metric = MulticlassF1Score(
            num_classes=self.dm.get_num_labels(), average="macro"
        ).to(self.device)

        # monitor
        self.monitor = self.config.callbacks.monitor

        # model-saving options
        self.version = 0
        while True:
            ckpt_dir = "model_save"
            if not os.path.exists(ckpt_dir):
                os.mkdir(ckpt_dir)

            self.save_path = os.path.join(ckpt_dir, f"version-{self.model_name}-{self.version}")
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
                break
            else:
                self.version += 1
        self.summarywriter = SummaryWriter(self.save_path)

        self.global_step = 0
        self.global_val_loss = 1e5
        self.global_val_acc = 0
        self.eval_step = 20
        logging.basicConfig(
            filename=os.path.join(self.save_path, "experiment.log"),
            level=logging.INFO,
            format="%(asctime)s > %(message)s",
        )

        # experiment-logging options
        self.best_result = {"version": self.version}

    def configure_optimizers(self, lr: float):
        # optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        return optimizer

    def save_checkpoint(
        self,
        epoch: int,
        value: float,
        model: nn.Module,
        remove_prev: bool = False,
        monitor: str = "val_loss",
    ) -> None:
        if monitor == "val_acc":
            logging.info(
                f"Val acc increased ({self.global_val_acc:.4f} → {value:.4f}). Saving model ..."
            )
            self.global_val_acc = value
        elif monitor == "val_loss":
            logging.info(
                f"Val loss decreased ({self.global_val_loss:.4f} → {value:.4f}). Saving model ..."
            )
            self.global_val_loss = value

        new_path = os.path.join(
            self.save_path, f"{self.model_name}_epoch_{epoch}_{monitor}_{value:.4f}.pt"
        )

        if remove_prev:
            for filename in glob.glob(os.path.join(self.save_path, "*.pt")):
                os.remove(filename)  # remove old checkpoint

        self.model.module.save_pretrained(new_path)

    def fit(self) -> dict:
        for epoch in tqdm(range(self.config.train.num_epochs), desc="epoch"):
            result = self._train_epoch(epoch)

            # update checkpoint
            if self.monitor == "val_acc":
                if result["val_acc"] > self.global_top1_acc:
                    self.save_checkpoint(epoch, result["val_acc"], self.model, monitor=self.monitor)
            elif self.monitor == "val_loss":
                if result["val_loss"] < self.global_val_loss:
                    self.save_checkpoint(
                        epoch, result["val_loss"], self.model, monitor=self.monitor
                    )

        self.summarywriter.close()
        return self.version

    def _train_epoch(self, epoch: int) -> dict:
        train_loss = AverageMeter()

        self.model.train()
        for step, batch in tqdm(
            enumerate(self.train_loader),
            desc="train_steps",
            total=len(self.train_loader),
        ):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            self.optimizer.zero_grad()
            if self.config.amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

            train_loss.update(loss.item())

            self.global_step += 1
            if self.global_step % self.eval_step == 0:
                logging.info(
                    f"[DP Version {self.version} Epoch {epoch}] global step: {self.global_step}, train loss: {loss.item():.3f}"
                )

        train_loss = train_loss.avg
        val_result = self.validate(epoch)

        # tensorboard writing
        val_loss = val_result["val_loss"]
        val_acc = val_result["val_acc"]
        val_macro_f1 = val_result["val_macro_f1"]
        val_weighted_f1 = val_result["val_weighted_f1"]

        self.summarywriter.add_scalars(
            "loss/step", {"val": val_loss, "train": train_loss}, self.global_step
        )
        self.summarywriter.add_scalars("loss/epoch", {"val": val_loss, "train": train_loss}, epoch)
        self.summarywriter.add_scalars("acc/epoch", {"val": val_acc}, epoch)
        self.summarywriter.add_scalars("macro_f1/epoch", {"val": val_macro_f1}, epoch)
        self.summarywriter.add_scalars("weighted_f1/epoch", {"val": val_weighted_f1}, epoch)
        logging.info(
            f"** global step: {self.global_step}, val loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, val_mcaro_f1: {val_macro_f1:.4f}, val_weighted_f1: {val_weighted_f1:.4f}"
        )

        return val_result

    def validate(self, epoch: int) -> Dict:
        val_loss = AverageMeter()
        val_acc = AverageMeter()
        val_macro_f1 = AverageMeter()
        val_weighted_f1 = AverageMeter()

        self.model.eval()
        with torch.no_grad():
            for step, batch in tqdm(
                enumerate(self.val_loader),
                desc="valid_steps",
                total=len(self.val_loader),
            ):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                val_loss.update(loss.item())

                acc = self.acc_metric(outputs.logits, batch["labels"])
                macro_f1 = self.macro_f1_metric(outputs.logits, batch["labels"])
                weighted_f1 = self.weighted_f1_metric(outputs.logits, batch["labels"])

                val_acc.update(acc.item())
                val_macro_f1.update(macro_f1.item())
                val_weighted_f1.update(weighted_f1.item())

        return {
            "val_loss": val_loss.avg,
            "val_acc": val_acc.avg,
            "val_macro_f1": val_macro_f1.avg,
            "val_weighted_f1": val_weighted_f1.avg,
        }
