# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Iterable

# --- stats / signals / models ---
from scipy.stats import skew, kurtosis
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.signal import butter, filtfilt, savgol_filter
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler,
    QuantileTransformer, PowerTransformer
)
from sklearn.mixture import GaussianMixture
import joblib

# ---------------------------
# 1) TRANSFORMS (stateless + helpers)
# ---------------------------
class Transforms:
    """Stateless transforms + safe helpers for sklearn-based steps."""

    @staticmethod
    def detrend_linear(y: np.ndarray) -> np.ndarray:
        t = np.arange(len(y))
        coef = np.polyfit(t, y, 1)
        return y - np.polyval(coef, t)

    @staticmethod
    def diff(y: np.ndarray, d: int = 1) -> np.ndarray:
        z = y
        for _ in range(int(d)):
            if len(z) < 2:
                return z
            z = np.diff(z, n=1)
        return z

    @staticmethod
    def highpass(y: np.ndarray, fs: float, fc: float = 1.0, order: int = 4) -> np.ndarray:
        nyq = 0.5 * fs
        fc = min(fc, nyq * 0.99)
        b, a = butter(order, fc / nyq, btype="highpass")
        return filtfilt(b, a, y)

    @staticmethod
    def savgol(y: np.ndarray, win: int = 31, poly: int = 2) -> np.ndarray:
        win = int(win) | 1
        poly = int(poly)
        win = max(win, poly + 2 | 1)
        if len(y) <= win:
            return y
        return savgol_filter(y, win, poly)

    @staticmethod
    def zscore_builtin(y: np.ndarray) -> np.ndarray:
        std = np.std(y) or 1.0
        return (y - np.mean(y)) / std

    @staticmethod
    def robust_scale_builtin(y: np.ndarray) -> np.ndarray:
        q75, q25 = np.percentile(y, [75, 25])
        iqr = q75 - q25
        iqr = iqr if iqr != 0 else 1.0
        return (y - np.median(y)) / iqr

    # ---- sklearn helpers ----
    @staticmethod
    def make_transformer(name: str, params: Dict[str, Any], n_samples: int) -> Optional[Any]:
        if name == "standard_scaler":
            return StandardScaler(with_mean=params.get("with_mean", True),
                                  with_std=params.get("with_std", True))
        if name == "robust_scaler":
            return RobustScaler(
                with_centering=params.get("with_centering", True),
                with_scaling=params.get("with_scaling", True),
                quantile_range=params.get("quantile_range", (25.0, 75.0))
            )
        if name == "minmax":
            return MinMaxScaler(feature_range=params.get("feature_range", (0.0, 1.0)))
        if name == "maxabs":
            return MaxAbsScaler()
        if name == "quantile":
            subsample = int(params.get("subsample", 100_000))
            nq_frac   = float(params.get("nq_frac", 1.0))
            # clamp n_quantiles to what we will actually fit on
            n_q = max(10, min(int(n_samples * nq_frac), subsample, n_samples))
            return QuantileTransformer(
                n_quantiles=n_q,
                output_distribution=params.get("output_distribution", "normal"),
                subsample=subsample,
                random_state=params.get("random_state", 0),
                copy=True
            )
        if name == "power_transform":
            method = params.get("method", "yeo-johnson")  # or 'box-cox'
            return PowerTransformer(method=method, standardize=params.get("standardize", True))
        return None

    @staticmethod
    def fit_apply_transformer_1d(y: np.ndarray, tr: Any) -> np.ndarray:
        """Fit a sklearn transformer on y (1D) and return transformed 1D result."""
        Y = y.reshape(-1, 1)
        # Box-Cox requires positivity
        if isinstance(tr, PowerTransformer) and tr.method == "box-cox" and np.min(Y) <= 0:
            Y = Y - np.min(Y) + np.finfo(float).eps
        yt = tr.fit_transform(Y).ravel()
        if not np.all(np.isfinite(yt)):
            return y
        if np.std(yt) == 0 and np.std(y) > 0:  # avoid collapse
            return y
        return yt

# ---------------------------
# 2) METRIC CLASSES
# ---------------------------
class Metric:
    """Base class for objective terms."""
    name: str = "metric"
    weight: float = 1.0
    def __call__(self, y: np.ndarray, x0: Optional[np.ndarray] = None) -> float:
        raise NotImplementedError

@dataclass
class SkewAbs(Metric):
    weight: float = 1.0
    name: str = "skew_abs"
    def __call__(self, y: np.ndarray, x0: Optional[np.ndarray] = None) -> float:
        return self.weight * abs(skew(y, bias=False))

@dataclass
class JarqueBeraPenalty(Metric):
    weight: float = 0.5
    p_ref: float = 0.05
    name: str = "jb_penalty"
    def __call__(self, y: np.ndarray, x0: Optional[np.ndarray] = None) -> float:
        _, p, _, _ = jarque_bera(y)
        return self.weight * max(0.0, self.p_ref - float(p))

@dataclass
class KpssPenalty(Metric):
    weight: float = 1.0
    p_ref: float = 0.05   # KPSS null = stationary; p<=0.05 suggests non-stationarity
    regression: str = "c"
    name: str = "kpss_penalty"
    def __call__(self, y: np.ndarray, x0: Optional[np.ndarray] = None) -> float:
        try:
            _, p, _, _ = kpss(y, regression=self.regression, nlags="auto")
            return self.weight * max(0.0, self.p_ref - float(p))
        except Exception:
            return self.weight  # conservative penalty if it fails

@dataclass
class LjungBoxPenalty(Metric):
    weight: float = 0.5
    lags: Tuple[int, ...] = (10, 20)
    p_ref: float = 0.05
    name: str = "lb_penalty"
    def __call__(self, y: np.ndarray, x0: Optional[np.ndarray] = None) -> float:
        pv = acorr_ljungbox(y, lags=self.lags, return_df=True)["lb_pvalue"].values
        return self.weight * float(np.sum(np.clip(self.p_ref - pv, 0, None)))

@dataclass
class TotalVariation(Metric):
    weight: float = 0.2
    name: str = "total_variation"
    def __call__(self, y: np.ndarray, x0: Optional[np.ndarray] = None) -> float:
        if len(y) < 2:
            return 0.0
        return self.weight * float(np.mean(np.abs(np.diff(y))))

@dataclass
class VarianceDrift(Metric):
    weight: float = 0.2
    name: str = "variance_drift"
    def __call__(self, y: np.ndarray, x0: Optional[np.ndarray] = None) -> float:
        if x0 is None:
            return 0.0
        v0 = np.var(x0) + 1e-12
        v1 = np.var(y) + 1e-12
        return self.weight * abs(float(np.log(v1 / v0)))

# --- Bimodality (configurable) ---
def _ashman_D(mu1, mu2, s1, s2) -> float:
    return np.sqrt(2.0) * np.abs(mu1 - mu2) / np.sqrt(s1**2 + s2**2)

@dataclass
class BimodalityTerm(Metric):
    """Configurable bimodality term. mode: 'discourage' | 'encourage' | 'ignore'"""
    weight: float = 0.5
    mode: str = "discourage"
    method: str = "gmm"  # 'gmm' | 'bc'
    min_D: float = 2.0
    min_weight: float = 0.05
    bc_ref: float = 0.555
    random_state: int = 0
    name: str = "bimodality"

    def __call__(self, y: np.ndarray, x0: Optional[np.ndarray] = None) -> float:
        if self.mode == "ignore" or self.weight <= 0:
            return 0.0
        if self.method == "gmm":
            x = y.reshape(-1, 1)
            g1 = GaussianMixture(1, covariance_type="full", random_state=self.random_state).fit(x)
            g2 = GaussianMixture(2, covariance_type="full", random_state=self.random_state).fit(x)
            delta_bic = g1.bic(x) - g2.bic(x)  # >0 favors 2 components
            mu = g2.means_.ravel()
            s = np.sqrt(np.array([g2.covariances_[k].ravel()[0] for k in range(2)]))
            w = g2.weights_
            D = _ashman_D(mu[0], mu[1], s[0], s[1])
            well_sep = (D > self.min_D) and np.all(w >= self.min_weight)
            strength = max(0.0, float(delta_bic)) * max(0.0, D - self.min_D) if well_sep else 0.0
        elif self.method == "bc":
            g1 = skew(y, bias=False)
            kP = kurtosis(y, fisher=False, bias=False)
            if kP <= 0:
                strength = 0.0
            else:
                bc = (g1**2 + 1.0) / kP
                strength = max(0.0, float(bc - self.bc_ref))
        else:
            raise ValueError(f"Unknown bimodality method: {self.method}")

        return (-self.weight * strength) if self.mode == "encourage" else (self.weight * strength)

# ---------------------------
# 3) OBJECTIVE
# ---------------------------
@dataclass
class Objective:
    """Aggregates metric values into a scalar loss."""
    metrics: Sequence[Metric]
    min_len: int = 32  # guard against over-differencing

    def evaluate(self, y: np.ndarray, x0: Optional[np.ndarray] = None) -> float:
        if y.size < self.min_len:
            return 1e3  # heavy penalty if signal collapses
        return float(np.sum([m(y, x0) for m in self.metrics]))

    def evaluate_signals(self, ys: List[np.ndarray], xs0: Optional[List[np.ndarray]] = None) -> float:
        losses = []
        if xs0 is None:
            xs0 = [None] * len(ys)
        for y, x0 in zip(ys, xs0):
            losses.append(self.evaluate(y, x0))
        return float(np.mean(losses)) if losses else 0.0

# ---------------------------
# 4) PIPELINE EXECUTION
# ---------------------------
@dataclass
class FittedStep:
    name: str
    params: Dict[str, Any]
    transformer: Optional[Any]  # sklearn transformer or None for stateless

class PipelineExecutor:
    """
    Fits and applies a pipeline defined as a list of steps: [{"name":..., "params":...}, ...]
    - Stateless steps are applied in-place.
    - Sklearn steps are fitted on the concatenation of current training signals, then applied.
    """

    def __init__(self, steps_config: List[Dict[str, Any]], fs: float):
        self.steps_config = steps_config
        self.fs = fs
        self.fitted_steps: List[FittedStep] = []

    # ---- fitting on training signals ----
    def fit(self, train_signals: List[np.ndarray]) -> "PipelineExecutor":
        currents = [np.asarray(s, dtype=float).ravel() for s in train_signals]
        self.fitted_steps = []

        for step in self.steps_config:
            name = step["name"]
            p = step.get("params", {})
            tr = Transforms.make_transformer(name, p, n_samples=sum(len(s) for s in currents))

            if tr is None:
                # stateless
                if name == "detrend_linear":
                    currents = [Transforms.detrend_linear(s) for s in currents]
                elif name == "diff":
                    currents = [Transforms.diff(s, d=int(p.get("d", 1))) for s in currents]
                elif name == "highpass":
                    currents = [Transforms.highpass(s, fs=self.fs,
                                                    fc=float(p.get("fc", 1.0)),
                                                    order=int(p.get("order", 4))) for s in currents]
                elif name == "savgol":
                    currents = [Transforms.savgol(s,
                                                  win=int(p.get("win", 31)),
                                                  poly=int(p.get("poly", 2))) for s in currents]
                elif name == "zscore_builtin":
                    currents = [Transforms.zscore_builtin(s) for s in currents]
                elif name == "robust_scale_builtin":
                    currents = [Transforms.robust_scale_builtin(s) for s in currents]
                else:
                    raise ValueError(f"Unknown stateless step: {name}")
                self.fitted_steps.append(FittedStep(name=name, params=p, transformer=None))
            else:
                # sklearn step: fit on concatenated data
                X = np.concatenate([s.reshape(-1, 1) for s in currents], axis=0)
                if isinstance(tr, PowerTransformer) and tr.method == "box-cox" and np.min(X) <= 0:
                    X = X - np.min(X) + np.finfo(float).eps
                tr.fit(X)
                # transform each current
                new_currents = []
                for s in currents:
                    Z = s.reshape(-1, 1)
                    if isinstance(tr, PowerTransformer) and tr.method == "box-cox" and np.min(Z) <= 0:
                        Z = Z - np.min(Z) + np.finfo(float).eps
                    s2 = tr.transform(Z).ravel()
                    new_currents.append(s2)
                currents = new_currents
                self.fitted_steps.append(FittedStep(name=name, params=p, transformer=tr))

        return self

    # ---- single signal transform ----
    def transform(self, signal: np.ndarray) -> np.ndarray:
        if not self.fitted_steps:
            raise RuntimeError("Pipeline not fitted. Call .fit(train_signals) first.")
        y = np.asarray(signal, dtype=float).ravel()
        for st in self.fitted_steps:
            if st.transformer is None:
                if st.name == "detrend_linear":
                    y = Transforms.detrend_linear(y)
                elif st.name == "diff":
                    y = Transforms.diff(y, d=int(st.params.get("d", 1)))
                elif st.name == "highpass":
                    y = Transforms.highpass(y, fs=self.fs,
                                            fc=float(st.params.get("fc", 1.0)),
                                            order=int(st.params.get("order", 4)))
                elif st.name == "savgol":
                    y = Transforms.savgol(y,
                                          win=int(st.params.get("win", 31)),
                                          poly=int(st.params.get("poly", 2)))
                elif st.name == "zscore_builtin":
                    y = Transforms.zscore_builtin(y)
                elif st.name == "robust_scale_builtin":
                    y = Transforms.robust_scale_builtin(y)
                else:
                    raise ValueError(f"Unknown stateless step: {st.name}")
            else:
                Z = y.reshape(-1, 1)
                if isinstance(st.transformer, PowerTransformer) and st.transformer.method == "box-cox" and np.min(Z) <= 0:
                    Z = Z - np.min(Z) + np.finfo(float).eps
                y = st.transformer.transform(Z).ravel()
        return y

    def transform_batch(self, signals: List[np.ndarray]) -> List[np.ndarray]:
        return [self.transform(s) for s in signals]

    # ---- persistence ----
    def save(self, path: str) -> None:
        joblib.dump(dict(steps_config=self.steps_config, fs=self.fs, fitted_steps=self.fitted_steps), path)

    @classmethod
    def load(cls, path: str) -> "PipelineExecutor":
        obj = joblib.load(path)
        pe = cls(obj["steps_config"], obj["fs"])
        pe.fitted_steps = obj["fitted_steps"]
        return pe

# ---------------------------
# 5) SAMPLER (Optuna-compatible)
# ---------------------------
def sample_steps(trial, fs: float, max_steps: int = 4) -> List[Dict[str, Any]]:
    """
    Returns a list of {"name": str, "params": dict} sampled for one trial.
    Make sure the consumer can execute these names in PipelineExecutor.
    """
    catalog = [
        # sklearn
        "standard_scaler", "robust_scaler", "minmax", "maxabs", "quantile", "power_transform",
        # stateless
        "detrend_linear", "diff", "highpass", "savgol", "zscore_builtin", "robust_scale_builtin"
    ]

    n_steps = trial.suggest_int("n_steps", 1, max_steps)
    steps: List[Dict[str, Any]] = []

    for i in range(n_steps):
        name = trial.suggest_categorical(f"step{i}_name", catalog)
        p: Dict[str, Any] = {}

        # sklearn
        if name == "standard_scaler":
            p["with_mean"] = trial.suggest_categorical(f"std_mean{i}", [True, False])
            p["with_std"]  = trial.suggest_categorical(f"std_std{i}", [True, False])

        elif name == "robust_scaler":
            qlo  = trial.suggest_float(f"rob_qlo{i}", 5.0, 30.0)
            qhi  = trial.suggest_float(f"rob_qhi{i}", 70.0, 95.0)
            if qhi <= qlo: qhi = qlo + 5.0
            p["quantile_range"] = (qlo, qhi)
            p["with_centering"] = trial.suggest_categorical(f"rob_ctr{i}", [True, False])
            p["with_scaling"]   = trial.suggest_categorical(f"rob_scl{i}", [True, False])

        elif name == "minmax":
            p["feature_range"] = trial.suggest_categorical(f"mm_range{i}", [(0.0, 1.0), (-1.0, 1.0)])

        elif name == "maxabs":
            pass

        elif name == "quantile":
            p["output_distribution"] = trial.suggest_categorical(f"qt_out{i}", ["normal", "uniform"])
            p["nq_frac"] = trial.suggest_float(f"qt_frac{i}", 0.1, 1.0)
            p["subsample"] = trial.suggest_int(f"qt_sub{i}", 20_000, 200_000, step=20_000)
            p["random_state"] = 0

        elif name == "power_transform":
            p["method"] = trial.suggest_categorical(f"pt_meth{i}", ["yeo-johnson", "box-cox"])
            p["standardize"] = trial.suggest_categorical(f"pt_std{i}", [True, False])

        # stateless
        elif name == "detrend_linear":
            pass
        elif name == "diff":
            p["d"] = trial.suggest_int(f"diff_d{i}", 0, 2)
        elif name == "highpass":
            max_fc = fs * 0.45
            p["fc"] = trial.suggest_float(f"hp_fc{i}", 0.1, max_fc)
            p["order"] = trial.suggest_int(f"hp_ord{i}", 2, 6)
        elif name == "savgol":
            p["win"]  = trial.suggest_int(f"sg_win{i}", 11, 151, step=2)
            p["poly"] = trial.suggest_int(f"sg_poly{i}", 2, 5)
        elif name == "zscore_builtin":
            pass
        elif name == "robust_scale_builtin":
            pass

        steps.append({"name": name, "params": p})

    return steps

# ---------------------------
# 6) EXAMPLE OBJECTIVE CONFIG
# ---------------------------
def default_objective() -> Objective:
    """
    A sensible default objective:
      - minimize skew
      - encourage stationarity (KPSS penalty)
      - reduce autocorrelation (Ljung-Box penalty)
      - regularize roughness and variance drift
      - discourage bimodality (tuneable)
    """
    metrics: List[Metric] = [
        SkewAbs(weight=1.0),
        JarqueBeraPenalty(weight=0.5, p_ref=0.05),
        KpssPenalty(weight=1.0, p_ref=0.05, regression="c"),
        LjungBoxPenalty(weight=0.5, lags=(10, 20), p_ref=0.05),
        TotalVariation(weight=0.2),
        VarianceDrift(weight=0.2),
        BimodalityTerm(weight=0.5, mode="discourage", method="gmm", min_D=2.0, min_weight=0.05)
    ]
    return Objective(metrics=metrics, min_len=32)

# ---------------------------
# 7) OPTIMIZATION LOOP (sketch)
# ---------------------------
# Example with Optuna (not imported here on purpose):
# import optuna
#
# def objective_fn(trial, signals: List[np.ndarray], fs: float) -> float:
#     steps = sample_steps(trial, fs=fs, max_steps=4)
#     trial.set_user_attr("steps_config", steps)
#     # Fit pipeline on training subset (no leakage)
#     pe = PipelineExecutor(steps_config=steps, fs=fs).fit(train_signals=signals)
#     ys = pe.transform_batch(signals)
#     obj = default_objective()
#     return obj.evaluate_signals(ys, xs0=signals)
#
# study = optuna.create_study(direction="minimize")
# study.optimize(lambda t: objective_fn(t, signals=my_train_signals, fs=12000), n_trials=200)
#
# best_steps = study.best_trial.user_attrs["steps_config"]
# pe = PipelineExecutor(best_steps, fs=12000).fit(train_signals=my_train_signals)
# y_new = pe.transform(x_new)
# pe.save("best_preproc.joblib")
# pe2 = PipelineExecutor.load("best_preproc.joblib")