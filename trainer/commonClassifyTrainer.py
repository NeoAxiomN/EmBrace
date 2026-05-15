from .trainer import baseTrainer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score, 
    f1_score, 
    cohen_kappa_score,
    precision_recall_curve, 
    auc, 
    roc_auc_score,
    confusion_matrix,
)
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class commonMultiClassTrainer(baseTrainer):
    def step(self, batch):
        x, y = batch[0].to(self.device), batch[1].to(self.device)
        logits = self.model(x)
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        preds = torch.argmax(logits, dim=-1).view(-1)
        labels = y.detach().view(-1)
        return {
            "loss": loss,
            "preds": preds,
            "labels": labels,
        }

    def summarize(self, tracker):
        all_preds = np.concatenate(tracker["preds"], axis=0)
        all_labels = np.concatenate(tracker["labels"], axis=0)
        
        return {
            "loss": np.mean(tracker["loss"]),
            "acc": accuracy_score(all_labels, all_preds),
            "b_acc": balanced_accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds, average="weighted"),
            "kappa": cohen_kappa_score(all_labels, all_preds),
            "cm": confusion_matrix(all_labels, all_preds).tolist(),
        }

    def _setup_criterion(self):
        return nn.CrossEntropyLoss(label_smoothing=self.hparams.label_smoothing)

    def _setup_scheduler(self):
        return CosineAnnealingLR(
            self.optimizer, T_max=self.hparams.epochs * len(self.data_loaders["train"]), eta_min=1e-6
        )
    
    def _setup_optimizer(self):
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": self.hparams.lr_backbone
            },
        ]
        return torch.optim.AdamW(
            params=param_dicts,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            eps=self.hparams.eps,
            betas=tuple(self.hparams.betas),
        )

    def loss_fn(self, logits, labels):
        return self.criterion(logits, labels)
    

    

class commonBinaryClassTrainer(commonMultiClassTrainer):
    def _setup_criterion(self):
        return nn.BCEWithLogitsLoss()

    def step(self, batch):
        x, y = batch[0].to(self.device), batch[1].to(self.device)
        logits = self.model(x) 

        if logits.ndim > 1 and logits.size(1) == 2:
            logits = logits[:, 1] - logits[:, 0]
        else:
            logits = logits.flatten()

        y_float = y.float().view_as(logits)
        loss = self.loss_fn(logits, y_float)

        probs = torch.sigmoid(logits).detach()

        return {
            "loss": loss,
            "probs": probs,
            "labels": y.detach(),
        }

    def summarize(self, tracker):
        avg_loss = np.mean(tracker["loss"])
        scores = np.concatenate(tracker["probs"])
        y_true = np.concatenate(tracker["labels"]).flatten()
        
        preds = (scores > 0.5).astype(int)

        precision, recall, _ = precision_recall_curve(y_true, scores, pos_label=1)

        return {
            "loss": avg_loss,
            "acc": accuracy_score(y_true, preds),
            "b_acc": balanced_accuracy_score(y_true, preds),
            "auc_pr": auc(recall, precision),
            "auroc": roc_auc_score(y_true, scores),
            "cm": confusion_matrix(y_true, preds).tolist(),
        }
    

