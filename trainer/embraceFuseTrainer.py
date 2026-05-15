from .commonClassifyTrainer import (
    commonMultiClassTrainer, 
    commonBinaryClassTrainer,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
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

class embraceFuseMultiClassTrainer(commonMultiClassTrainer):
    def __init__(
        self, 
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        task_model = self._unwrap_model()
        self.weight_ce = task_model.weight_ce
        self.weight_emb = task_model.weight_emb
        self.weight_attn = task_model.weight_attn
    
    def step(self, batch):
        x, y = batch[0].to(self.device), batch[1].to(self.device)
        m = batch[2].to(self.device) if len(batch) > 2 else None

        outputs = self.model(x, m)
        loss = self.loss_fn(outputs, y)
        preds = torch.argmax(outputs["logits"], dim=-1).view(-1)
        labels = y.detach().view(-1)
        ret = {
            "loss": loss["loss"],
            "preds": preds,
            "labels": labels,
        }
        if self.model.training:
            ret.update({
                "loss_attn": loss.get("loss_attn", 0),
                "loss_emb": loss.get("loss_emb", 0),
                "k_max": outputs["k_attn"].max(),
                "t_max": outputs["t_attn"].max(),
                "w_attn": outputs["w_attn"].detach().mean(dim=0),
                "w_emb": outputs["w_emb"].detach().mean(dim=0)
            })
            
        return ret
    
    def loss_fn(self, outputs, labels):
        nClasses = outputs["logits"].size(-1)
        loss_task = F.cross_entropy(
            outputs["logits"].view(-1, nClasses), 
            labels.view(-1),
            label_smoothing=self.hparams.label_smoothing
        )
        
        if self.model.training:
            loss_emb = F.mse_loss(outputs["t_emb"], outputs["k_emb"])
            loss_attn = F.mse_loss(outputs["t_attn"], outputs["k_attn"])
            
            total_loss = (
                self.weight_ce * loss_task +
                self.weight_emb * loss_emb +
                self.weight_attn * loss_attn
            )

            return {
                "loss": total_loss,
                "loss_task": loss_task.detach(),
                "loss_emb": loss_emb.detach(),
                "loss_attn": loss_attn.detach()
            }
        else:
            return {"loss": loss_task}

    def summarize(self, tracker):
        all_preds = np.concatenate(tracker["preds"], axis=0)
        all_labels = np.concatenate(tracker["labels"], axis=0)
        
        metrics = {
            "loss": np.mean(tracker["loss"]),
            "acc": accuracy_score(all_labels, all_preds),
            "b_acc": balanced_accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds, average="weighted"),
            "kappa": cohen_kappa_score(all_labels, all_preds),
            "cm": confusion_matrix(all_labels, all_preds).tolist(),
        }
        if "w_attn" in tracker:
            metrics["avg_w_attn"] = np.mean(tracker["w_attn"], axis=0).tolist()
            metrics["avg_w_emb"] = np.mean(tracker["w_emb"], axis=0).tolist()

        if "k_max" in tracker:
            metrics.update({
                "avg_k_max": np.mean(tracker["k_max"]),
                "avg_t_max": np.mean(tracker["t_max"]),
                "avg_loss_attn": np.mean(tracker["loss_attn"]),
                "avg_loss_emb": np.mean(tracker["loss_emb"]),
            })

        return metrics


class embraceFuseBinaryClassTrainer(commonBinaryClassTrainer):
    def __init__(
        self,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        task_model = self._unwrap_model()
        self.weight_ce = task_model.weight_ce
        self.weight_emb = task_model.weight_emb
        self.weight_attn = task_model.weight_attn

    def step(self, batch):
        x, y = batch[0].to(self.device), batch[1].to(self.device)
        m = batch[2].to(self.device) if len(batch) > 2 else None

        outputs = self.model(x, m)
        logits = outputs["logits"]
        if logits.ndim > 1 and logits.size(1) == 2:
            logits = logits[:, 1] - logits[:, 0]
        else:
            logits = logits.flatten()
        outputs["logits"] = logits

        y_float = y.float().view_as(logits)
        loss = self.loss_fn(outputs, y_float)
        probs = torch.sigmoid(logits).detach()
        labels = y.detach()
    
        ret = {
            "loss": loss["loss"],
            "probs": probs,
            "labels": labels,
        }
        if self.model.training:
            ret.update({
                "loss_attn": loss.get("loss_attn", 0),
                "loss_emb": loss.get("loss_emb", 0),
                "k_max": outputs["k_attn"].max(),
                "t_max": outputs["t_attn"].max(),
            })
            
        return ret
    

    def loss_fn(self, outputs, labels):
        loss_task = F.binary_cross_entropy_with_logits(outputs["logits"], labels)
        
        if self.model.training and "t_emb" in outputs:
            loss_emb = F.mse_loss(outputs["t_emb"], outputs["k_emb"])
            loss_attn = F.mse_loss(outputs["t_attn"], outputs["k_attn"])
            
            total_loss = (self.weight_ce * loss_task + 
                          self.weight_emb * loss_emb + 
                          self.weight_attn * loss_attn)
            return {
                "loss": total_loss,
                "loss_task": loss_task.detach(),
                "loss_emb": loss_emb.detach(),
                "loss_attn": loss_attn.detach()
            }
        else:
            return {"loss": loss_task}
    

    def summarize(self, tracker):
        scores = np.concatenate(tracker["probs"])
        y_true = np.concatenate(tracker["labels"]).flatten()
        
        preds = (scores > 0.5).astype(int)

        precision, recall, _ = precision_recall_curve(y_true, scores, pos_label=1)

        metrics = {
            "loss": np.mean(tracker["loss"]),
            "acc": accuracy_score(y_true, preds),
            "b_acc": balanced_accuracy_score(y_true, preds),
            "auc_pr": auc(recall, precision),
            "auroc": roc_auc_score(y_true, scores),
            "cm": confusion_matrix(y_true, preds).tolist(),
        }
    
        if "k_max" in tracker:
            metrics.update({
                "avg_k_max": np.mean(tracker["k_max"]),
                "avg_t_max": np.mean(tracker["t_max"]),
                "avg_loss_attn": np.mean(tracker["loss_attn"]),
                "avg_loss_emb": np.mean(tracker["loss_emb"]),
            })

        return metrics
    
