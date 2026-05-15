from .PARC import PARCEvaluator
from .utils import load_stratified_sampling_dataset

import time   
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import gc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class Selector(object):
    def __init__(
        self,
        num_classes,
        model_sources,
        train_dataloader,
        val_dataloader,
        batch_size,
        device,
        sample_ratio=0.2,
        cfg=None
    ):  
        self.model_sources = model_sources
        self.num_classes = num_classes
        self.train_data_loader = train_dataloader
        self.val_data_loader = val_dataloader
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.sample_ratio = sample_ratio
        self.cfg = cfg
        
    def _load_dataloader(self ):
        if self.sample_ratio == 1:
            dataloader = self.val_data_loader 
        elif self.sample_ratio == 0:
            dataloader = self.train_data_loader
        else:
            dataloader = load_stratified_sampling_dataset(
                num_classes=self.num_classes,
                batch_size=self.batch_size,
                train_data_loader=self.train_data_loader,
                val_data_loader=self.val_data_loader,
                sample_ratio=self.sample_ratio
            )

        return dataloader           

    def select_best_model(self):
        MODEL_MAP = {
            "cbramod": "CBraMod",
            "codebrain": "CodeBrain",
            "labram": "LaBraM",
            "csbrain": "CSBrain"
        }
        scores = {}
        dataloader = self._load_dataloader()
        evaluator = PARCEvaluator(n_dims=self.cfg.parc.n_dims)
        from model import efms 

        
        print(f"\n>> Starting Model Selection (Metric: PARC, Sample Ratio: {self.sample_ratio})")
        for name in self.model_sources:
            def get_model_class(name):
                cls_name = MODEL_MAP.get(name.lower(), name.capitalize())
                return getattr(efms, cls_name)

            modelClass = get_model_class(name)
            model = modelClass(
                dataset=self.cfg.dataset.name,
                num_classes=self.cfg.dataset.num_classes,
                dropout=self.cfg.model.dropout,
                foundation_dir=self.cfg.model.foundation_dir_dict[name],
                use_pretrained_weights=True
            )
            model = model.backbone.to(self.device)

            with torch.no_grad():
                model.eval()

                F_list, y_true_list = [], []
        
                for batch_x, batch_y in tqdm(dataloader, desc=f"Evaluating {name:10}", leave=False):
                    batch_x = batch_x.to(self.device)
                    if name == 'labram':
                        features = model.forward_features(batch_x) 
                    else:
                        features = model(batch_x) 

                    if features.ndim > 2:
                        torch.manual_seed(self.cfg.seed) 
        
                        N = features.shape[0]
                        features = features.reshape(N, -1).to(torch.float32)
                        D_in = features.shape[1]
                        projector = nn.Sequential(
                            nn.Linear(D_in, 800), 
                            nn.ELU(),
                            nn.Linear(800, 200),  
                            nn.ELU()
                        ).to(self.device)
                        # def init_weights(m):
                        #     if isinstance(m, nn.Linear):
                        #         nn.init.orthogonal_(m.weight)
                        #         if m.bias is not None:
                        #             nn.init.constant_(m.bias, 0)
                        # projector.apply(init_weights)
                        projector.eval()

                        feats = projector(features)

                    batch_y = batch_y.reshape(-1)
                    sp = features.shape

                    F_list.append(features.cpu().numpy())
                    y_true_list.append(batch_y.cpu().numpy())

                    del features, batch_x
                    torch.cuda.empty_cache()

        
            F_list = np.concatenate(F_list, axis=0)
            y_true_list = np.concatenate(y_true_list, axis=0)
            feats = F_list
            labels = y_true_list

                
            score = evaluator.compute_parc(feats, labels, self.num_classes)
            scores[name] = score

            print(f"Model: {name:10} | PARC Score: {score:.6f} | Features Shape: {sp}")

            model.cpu()
            del model, F_list, y_true_list, feats, labels

            torch.cuda.empty_cache()
            gc.collect()
            # time.sleep(2)
            
        valid_scores = {k: v for k, v in scores.items() if v != -np.inf}
        
        if not valid_scores:
            return {"best_model": "None", "reason": f"评估失败。", f"parc_scores": scores}
            
        best_model = max(valid_scores, key=valid_scores.get)

        return {
            "best_model": best_model,
            "reason": f"分数最高 ({valid_scores[best_model]:.6f})。",
            f"parc_scores": scores,
        }