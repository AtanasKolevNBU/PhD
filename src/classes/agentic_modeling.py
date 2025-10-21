# -*- coding: utf-8 -*-
from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import type_of_target

# XGBoost is optional
try:
    from xgboost import XGBClassifier
    _HAVE_XGB = True
except Exception:
    print("XGBoost Unavailable!")
    _HAVE_XGB = False
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import optuna
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================
# 1. PyTorch ANN wrapper
# =====================================
class TorchANN(nn.Module):
    def __init__(self, input_dim, hidden_sizes=(128,64), dropout=0.2, num_classes=2):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        out_dim = num_classes if num_classes > 2 else 1
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        self.num_classes = num_classes
    def forward(self, x): return self.net(x)

class TorchANNClassifier:
    """
    Sklearn-like wrapper with:
    - early stopping (val split + patience)
    - per-epoch logging
    - time budget (max_seconds_per_fit)
    - Optuna pruning hook (trial optional)
    """
    def __init__(self,
                 input_dim=None,
                 hidden_sizes=(128,64),
                 dropout=0.2,
                 lr=1e-3,
                 batch_size=128,
                 epochs=60,
                 patience=8,
                 val_frac=0.15,
                 num_classes=2,
                 device=DEVICE,
                 num_workers=0,
                 pin_memory=None,
                 use_amp=None,
                 max_seconds_per_fit=None,   # e.g. 60
                 verbose=1):
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.val_frac = val_frac
        self.num_classes = num_classes
        self.device = device
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory) if pin_memory is not None else (self.device=="cuda")
        self.use_amp = bool(use_amp) if use_amp is not None else (self.device=="cuda")
        self.max_seconds_per_fit = max_seconds_per_fit
        self.verbose = verbose

        self.model = None
        self.scaler = StandardScaler()

    def _build_model(self, input_dim):
        self.model = TorchANN(input_dim, self.hidden_sizes, self.dropout, self.num_classes).to(self.device)

    def _criterion(self):
        return nn.CrossEntropyLoss() if self.num_classes > 2 else nn.BCEWithLogitsLoss()

    def _make_loaders(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        X = self.scaler.fit_transform(X)
        if self.input_dim is None:
            self.input_dim = X.shape[1]
        ds = TensorDataset(torch.tensor(X), torch.tensor(y))
        if self.val_frac and self.val_frac > 0.0:
            n = len(ds)
            n_val = max(1, int(n * self.val_frac))
            n_tr = n - n_val
            dtr, dva = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(0))
        else:
            dtr, dva = ds, None
        dl_tr = DataLoader(dtr, batch_size=self.batch_size, shuffle=True,
                           num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=False)
        dl_va = None
        if dva is not None:
            dl_va = DataLoader(dva, batch_size=self.batch_size, shuffle=False,
                               num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=False)
        return dl_tr, dl_va

    def fit(self, X, y, trial=None):
        start = time.time()
        dl_tr, dl_va = self._make_loaders(X, y)
        self._build_model(self.input_dim)
        criterion = self._criterion()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scaler_amp = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        best_va = np.inf
        best_state = None
        patience_left = self.patience

        for epoch in range(1, self.epochs+1):
            self.model.train()
            ep_loss = 0.0
            for xb, yb in dl_tr:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    out = self.model(xb)
                    if self.num_classes > 2:
                        loss = criterion(out, yb)
                    else:
                        loss = criterion(out.view(-1), yb.float())
                scaler_amp.scale(loss).backward()
                scaler_amp.step(optimizer)
                scaler_amp.update()
                ep_loss += float(loss.detach().cpu())

            # validation
            va_loss = np.nan
            if dl_va is not None:
                self.model.eval()
                total = 0.0
                with torch.no_grad():
                    for xb, yb in dl_va:
                        xb = xb.to(self.device, non_blocking=True)
                        yb = yb.to(self.device, non_blocking=True)
                        with torch.cuda.amp.autocast(enabled=self.use_amp):
                            out = self.model(xb)
                            if self.num_classes > 2:
                                l = criterion(out, yb)
                            else:
                                l = criterion(out.view(-1), yb.float())
                        total += float(l.detach().cpu())
                va_loss = total / max(1, len(dl_va))

                # early stopping
                if va_loss + 1e-7 < best_va:
                    best_va = va_loss
                    best_state = {k: v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
                    patience_left = self.patience
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        if self.verbose:
                            print(f"[TorchANN] Early stop at epoch {epoch} (best val loss {best_va:.4f})", flush=True)
                        break

                # Optuna pruning
                if trial is not None:
                    trial.report(-va_loss, step=epoch)  # higher is better → use negative loss
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            if self.verbose and (epoch == 1 or epoch % 5 == 0):
                print(f"[TorchANN] epoch {epoch:03d} train_loss={ep_loss/len(dl_tr):.4f} val_loss={va_loss:.4f}", flush=True)

            # time budget
            if self.max_seconds_per_fit is not None and (time.time() - start) > self.max_seconds_per_fit:
                if self.verbose:
                    print(f"[TorchANN] Stopping due to time budget ({self.max_seconds_per_fit}s)", flush=True)
                break

        # restore best
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # sync to ensure no hidden CUDA work remains
        if self.device == "cuda":
            torch.cuda.synchronize()

        return self

    def predict_proba(self, X):
        X = self.scaler.transform(np.asarray(X, dtype=np.float32))
        self.model.eval()
        with torch.no_grad():
            xb = torch.tensor(X).to(self.device, non_blocking=True)
            logits = self.model(xb)
            if self.num_classes > 2:
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            else:
                probs1 = torch.sigmoid(logits.view(-1)).cpu().numpy()
                probs = np.vstack([1 - probs1, probs1]).T
        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

# ============== Metric classes ==============
class ClfMetric:
    name: str = "metric"
    weight: float = 1.0
    average: str = "binary"   # 'binary' | 'macro' | 'weighted'
    def __call__(self, y_true, y_pred, y_proba=None) -> float:
        raise NotImplementedError

@dataclass
class AccuracyMetric(ClfMetric):
    name: str = "accuracy"
    weight: float = 1.0
    def __call__(self, y_true, y_pred, y_proba=None) -> float:
        return self.weight * accuracy_score(y_true, y_pred)

@dataclass
class PrecisionMetric(ClfMetric):
    name: str = "precision"
    weight: float = 1.0
    average: str = "binary"
    def __call__(self, y_true, y_pred, y_proba=None) -> float:
        return self.weight * precision_score(y_true, y_pred, average=self.average, zero_division=0)

@dataclass
class RecallMetric(ClfMetric):
    name: str = "recall"
    weight: float = 1.0
    average: str = "binary"
    def __call__(self, y_true, y_pred, y_proba=None) -> float:
        return self.weight * recall_score(y_true, y_pred, average=self.average, zero_division=0)

@dataclass
class F1Metric(ClfMetric):
    name: str = "f1"
    weight: float = 1.0
    average: str = "binary"
    def __call__(self, y_true, y_pred, y_proba=None) -> float:
        return self.weight * f1_score(y_true, y_pred, average=self.average, zero_division=0)

@dataclass
class RocAucMetric(ClfMetric):
    name: str = "roc_auc"
    weight: float = 1.0
    multi_class: str = "ovr"  # 'ovr' or 'ovo' for multi-class
    average: Optional[str] = "weighted"  # only used for multi-class
    def __call__(self, y_true, y_pred, y_proba=None) -> float:
        if y_proba is None:
            return 0.0  # no probability -> can't compute ROC AUC
        target_type = type_of_target(y_true)
        if target_type == "binary":
            # y_proba can be (n_samples,) or (n_samples, 2)
            probs = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            return self.weight * roc_auc_score(y_true, probs)
        else:
            # multi-class
            return self.weight * roc_auc_score(
                y_true, y_proba, multi_class=self.multi_class, average=self.average
            )

# ============== Objective ==============
@dataclass
class ObjectiveClassifier:
    metrics: List[ClfMetric]
    greater_is_better: bool = True

    def aggregate(self, scores: Dict[str, float]) -> float:
        # sum of already weighted metrics
        total = float(sum(scores.values()))
        return total if self.greater_is_better else -total

# =====================================
# 2. Model Factory
# =====================================
class ModelFactory:
    @staticmethod
    def build(config, input_dim=None, num_classes=2):
        mtype = config["model_type"]
        params = config.get("params", {}).copy()

        if mtype == "svc":
            params.setdefault("probability", True)
            base = SVC(**params)
            return Pipeline([("scaler", StandardScaler()), ("svc", base)])

        elif mtype == "mlp":
            base = MLPClassifier(**params)
            return Pipeline([("scaler", StandardScaler()), ("mlp", base)])

        elif mtype == "torch":
            # Use PyTorch ANN
            params.setdefault("hidden_sizes", (128, 64))
            params.setdefault("dropout", 0.2)
            params.setdefault("lr", 1e-3)
            params.setdefault("batch_size", 64)
            params.setdefault("epochs", 30)
            return TorchANNClassifier(
                input_dim=input_dim,
                hidden_sizes=params["hidden_sizes"],
                dropout=params["dropout"],
                lr=params["lr"],
                batch_size=params["batch_size"],
                epochs=params["epochs"],
                num_classes=num_classes,
            )

        elif mtype == "xgb":
            if not _HAVE_XGB:
                raise RuntimeError("XGBoost not available")
            params.setdefault("eval_metric", "logloss")
            params.setdefault("tree_method", "hist")
            return XGBClassifier(**params)

        else:
            raise ValueError(f"Unknown model_type {mtype}")

# =====================================
# 3. Model Sampler for Optuna
# =====================================
class ModelSampler:
    @staticmethod
    def sample(trial, use_torch=True):
        models = ["svc", "mlp", "xgb"]
        if use_torch and DEVICE == "cuda":
            models.append("torch")
        model_type = trial.suggest_categorical("model_type", models)
        params = {}

        if model_type == "svc":
            params["C"] = trial.suggest_float("svc_C", 1e-3, 1e3, log=True)
            params["kernel"] = trial.suggest_categorical("svc_kernel", ["rbf", "poly"])
            params["gamma"] = trial.suggest_float("svc_gamma", 1e-4, 1.0, log=True)
            params["class_weight"] = trial.suggest_categorical("svc_cw", [None, "balanced"])

        elif model_type == "mlp":
            params["hidden_layer_sizes"] = trial.suggest_categorical("mlp_hidden", [(128,), (64, 64), (128, 64)])
            params["activation"] = trial.suggest_categorical("mlp_act", ["relu", "tanh"])
            params["alpha"] = trial.suggest_float("mlp_alpha", 1e-6, 1e-2, log=True)
            params["learning_rate_init"] = trial.suggest_float("mlp_lr", 1e-4, 1e-2, log=True)
            params["max_iter"] = trial.suggest_int("mlp_max_iter", 150, 400)

        elif model_type == "torch":
            params["hidden_sizes"] = trial.suggest_categorical("torch_hidden", [(128, 64), (64, 64, 32)])
            params["dropout"] = trial.suggest_float("torch_dropout", 0.0, 0.5)
            params["lr"] = trial.suggest_float("torch_lr", 1e-4, 1e-2, log=True)
            params["batch_size"] = trial.suggest_categorical("torch_batch", [32, 64, 128])
            params["epochs"] = trial.suggest_int("torch_epochs", 20, 80)

        elif model_type == "xgb":
            params["n_estimators"] = trial.suggest_int("xgb_n", 100, 400)
            params["max_depth"] = trial.suggest_int("xgb_depth", 3, 8)
            params["learning_rate"] = trial.suggest_float("xgb_lr", 1e-3, 0.3, log=True)

        return {"model_type": model_type, "params": params}

# ============== Cross-validated evaluation ==============
@dataclass
class CVEvaluator:
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 0

    def _get_proba(self, estimator, X) -> Optional[np.ndarray]:
        # Try predict_proba, else decision_function
        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(X)
            # Some estimators return shape (n_samples,) for binary; unify to (n,2)
            if proba.ndim == 1:
                proba = np.vstack([1 - proba, proba]).T
            return proba
        if hasattr(estimator, "decision_function"):
            dec = estimator.decision_function(X)
            # map decision scores to probabilities with a sigmoid-like scaling
            # Here, we simply return scores; roc_auc can accept scores
            if dec.ndim == 1:
                dec = dec.reshape(-1, 1)
            return dec
        return None

    def evaluate(self, estimator, X: np.ndarray, y: np.ndarray, objective: ObjectiveClassifier) -> Tuple[float, Dict[str, float]]:
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        metric_sums: Dict[str, float] = {}
        nfolds = 0

        for tr_idx, va_idx in skf.split(X, y):
            nfolds += 1
            Xtr, Xva = X[tr_idx], X[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est = estimator  # rebuild fresh each fold if pipeline with internal state?
                # Rebuild to avoid state leakage between folds
                est = ModelFactory.build({"model_type": estimator["__model_type__"],
                                          "params": estimator["__params__"]},
                                         random_state=self.random_state)
                est.fit(Xtr, ytr)
            y_pred = est.predict(Xva)
            y_proba = self._get_proba(est, Xva)

            # compute all metrics
            fold_scores: Dict[str, float] = {}
            for m in objective.metrics:
                val = m(yva, y_pred, y_proba)
                fold_scores[m.name] = fold_scores.get(m.name, 0.0) + float(val)

            # accumulate
            for k, v in fold_scores.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + v

        # average across folds
        metric_avgs = {k: v / nfolds for k, v in metric_sums.items()}
        aggregate = objective.aggregate(metric_avgs)
        return aggregate, metric_avgs

# ============== Study runner (Optuna) ==============
class StudyRunner:
    def __init__(self, objective: ObjectiveClassifier, cv: CVEvaluator):
        self.objective = objective
        self.cv = cv
        self.best_config: Optional[Dict[str, Any]] = None
        self.best_metrics: Optional[Dict[str, float]] = None
        self.best_score: Optional[float] = None

    def optuna_objective(trial, X, y, weights, n_splits=5):
        cfg = ModelSampler.sample(trial, use_torch=True)
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        scores = []
        for tr_idx, va_idx in kf.split(X, y):
            Xtr, Xva = X[tr_idx], X[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]

            if cfg["model_type"] == "torch" and DEVICE == "cuda":
                # bounded, prune-able fit
                model = TorchANNClassifier(
                    input_dim=X.shape[1], num_classes=len(np.unique(y)),
                    epochs=60, patience=8, max_seconds_per_fit=45, verbose=1
                )
                model.fit(Xtr, ytr, trial=trial)
            else:
                model = ModelFactory.build(cfg, input_dim=X.shape[1], num_classes=len(np.unique(y)))
                model.fit(Xtr, ytr)

            y_pred = model.predict(Xva)
            y_proba = model.predict_proba(Xva)
            m = {
                "accuracy": accuracy_score(yva, y_pred),
                "precision": precision_score(yva, y_pred, zero_division=0),
                "recall": recall_score(yva, y_pred, zero_division=0),
                "f1": f1_score(yva, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(yva, y_proba[:,1]) if y_proba is not None else 0.0
            }
            scores.append(sum(m[k]*weights.get(k,1.0) for k in m))
        return float(np.mean(scores))

    def run(self, X: np.ndarray, y: np.ndarray, weights:dict, n_trials: int = 50, direction: str = "maximize", seed: int = 0):
        import optuna
        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=seed), pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5))
        study.optimize(lambda t: self.optuna_objective(t, X, y,weights=weights, n_splits=5), n_trials=n_trials)

        self.best_config = study.best_trial.user_attrs["config"]
        self.best_metrics = {k.replace("metric_", ""): v for k, v in study.best_trial.user_attrs.items() if k.startswith("metric_")}
        self.best_score = study.best_value
        return study

# ============== Best model wrapper ==============
class BestModel:
    def __init__(self, config: Dict[str, Any], random_state: int = 0):
        self.config = config
        self.random_state = random_state
        self.estimator = ModelFactory.build(config, random_state=random_state)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.estimator.fit(X, y)
        return self

    def predict(self, X: np.ndarray):
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray):
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X)
        if hasattr(self.estimator, "decision_function"):
            dec = self.estimator.decision_function(X)
            if dec.ndim == 1:
                dec = dec.reshape(-1, 1)
            return dec
        return None

# =====================================
# 4. Evaluation Objective
# =====================================
def classification_objective(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba[:, 1]) if y_proba is not None else 0.0,
    }

def weighted_score(metrics: dict, weights: dict):
    return sum(metrics[k] * weights.get(k, 1.0) for k in metrics)

# =====================================
# 5. Optuna-compatible Objective
# =====================================
def optuna_objective(trial, X, y, weights=None, n_splits=5):
    from sklearn.model_selection import StratifiedKFold
    weights = weights or {"accuracy": 1, "precision": 1, "recall": 1, "f1": 1, "roc_auc": 1}

    cfg = ModelSampler.sample(trial)
    mtype = cfg["model_type"]
    params = cfg["params"]

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    fold_scores = []

    for train_idx, test_idx in kf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        if mtype == "torch":
            model = ModelFactory.build(cfg, input_dim=X.shape[1], num_classes=len(np.unique(y)))
        else:
            model = ModelFactory.build(cfg)

        model.fit(Xtr, ytr)
        y_pred = model.predict(Xte)
        y_proba = model.predict_proba(Xte)
        metrics = classification_objective(yte, y_pred, y_proba)
        fold_scores.append(weighted_score(metrics, weights))

    return float(np.mean(fold_scores))


# =====================================
# 6. Example Usage
# =====================================
"""
import optuna

X, y = ...  # numpy arrays
weights = {"accuracy": 0.3, "precision": 0.2, "recall": 0.2, "f1": 0.2, "roc_auc": 0.1}

study = optuna.create_study(direction="maximize")
study.optimize(lambda t: optuna_objective(t, X, y, weights, n_splits=5), n_trials=50)

print("Best score:", study.best_value)
print("Best config:", study.best_trial.params)
cfg = study.best_trial.user_attrs if "config" in study.best_trial.user_attrs else None
"""