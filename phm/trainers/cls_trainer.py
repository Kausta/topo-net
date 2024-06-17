from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv

import pytorch_lightning as pl
import torchmetrics
import torchmetrics.classification

import phm
import phm.nn as pnn
from phm.util.typing import *

from .base import BaseTrainer


@phm.register("classification-trainer")
class BinaryClassificationTrainer(BaseTrainer):
    @dataclass
    class Config(BaseTrainer.Config):
        model_type: str = ""
        model: dict = field(default_factory=dict)

        task: str = "binary" # "binary" | "multiclass" | "multilabel"
        num_classes: Optional[int] = None # Shouldn't be None for "multiclass"
        num_labels: Optional[int] = None # Shouldn't be None for "multilabel"

        betti_aux: bool = False
        pi_aux: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()

        self.model = phm.find(self.cfg.model_type)(self.cfg.model)
        phm.info(self.model)

        self.task = self.cfg.task
        self.num_classes = self.cfg.num_classes
        self.num_labels = self.cfg.num_labels
        if self.task == "binary":
            self.ce_loss = nn.BCEWithLogitsLoss()
            self.target_map = lambda t: t.float()
        elif self.task == "multiclass":
            assert self.num_classes is not None, "multiclass requires num_classes"
            self.ce_loss = nn.CrossEntropyLoss()
            self.target_map = lambda t: t.squeeze(-1)
        elif self.task == "multilabel":
            assert self.num_labels is not None, "multilabel requires num_labels"
            self.ce_loss = nn.BCEWithLogitsLoss()
            self.target_map = lambda t: t.float()
        else:
            raise ValueError(f"Unknown task {self.task}")        

        metric_kwargs = {
            "task": self.task,
            "num_classes": self.num_classes,
            "num_labels": self.num_labels
        }
        self.train_acc = torchmetrics.classification.Accuracy(**metric_kwargs)
        self.train_roc = torchmetrics.classification.AUROC(**metric_kwargs)

        self.val_acc = torchmetrics.classification.Accuracy(**metric_kwargs)
        self.val_roc = torchmetrics.classification.AUROC(**metric_kwargs)

        self.test_acc = torchmetrics.classification.Accuracy(**metric_kwargs)
        self.test_roc = torchmetrics.classification.AUROC(**metric_kwargs)

        self.betti_aux = self.cfg.betti_aux
        self.pi_aux = self.cfg.pi_aux
        assert not (self.betti_aux and self.pi_aux)

    def forward(self, x, input_dict):
        kwargs = {}
        if self.betti_aux:
            assert "betti" in input_dict
            kwargs["aux"] = input_dict["betti"]
        if self.pi_aux:
            assert "persistence_image" in input_dict
            kwargs["aux"] = input_dict["persistence_image"]
        pred, out = self.model(x, **kwargs)
        return pred, out

    def loss(self, input_dict, phase="train"):
        pred, other = self.forward(input_dict["input"], input_dict)
        target = self.target_map(input_dict["target"])

        out = {
            "pred": pred,
            "target": target,
            **other
        }

        loss_terms = {}

        loss_prefix = f"loss_"

        def set_loss(name, value):
            loss_terms[f"{loss_prefix}{name}"] = value

        # (Binary) Cross entropy
        set_loss("ce", self.ce_loss(pred, target))

        loss = 0.0
        on_step = (phase == "train")
        for name, value in loss_terms.items():
            self.log(f"{phase}/{name}", value, on_step=on_step, on_epoch=True)
            if name.startswith(loss_prefix):
                loss_weighted = value * self.C(
                    self.cfg.loss[name.replace(loss_prefix, "lambda_")]
                )
                self.log(f"{phase}/{name}_w", loss_weighted,
                         on_step=on_step, on_epoch=True)
                loss += loss_weighted

        if phase == "train":
            for name, value in self.cfg.loss.items():
                self.log(
                    f"train_params/{name}", self.C(value), on_step=True, on_epoch=False)

        self.log(f"{phase}/loss", loss, on_step=on_step,
                 on_epoch=True, prog_bar=True)

        return {"loss": loss, "output": out}

    def training_step(self, batch, batch_idx):
        out = self.loss(batch, "train")
        self.log_stats(out["output"], "train")
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self.loss(batch, "val")
        self.log_stats(out["output"], "val")

    def test_step(self, batch, batch_idx):
        out = self.loss(batch, "test")
        self.log_stats(out["output"], "test")

    def log_stats(self, output, phase="train"):
        acc_fn = getattr(self, f"{phase}_acc")
        roc_fn = getattr(self, f"{phase}_roc")

        target = output["target"]
        if self.task == "multilabel":
            target = target.long()
        acc_fn(output["pred"], target)
        roc_fn(output["pred"], target)

        self.log(f"{phase}/accuracy", acc_fn,
                 on_step=(phase == "train"), on_epoch=True)
        self.log(f"{phase}/roc_auc", roc_fn,
                 on_step=(phase == "train"), on_epoch=True)


