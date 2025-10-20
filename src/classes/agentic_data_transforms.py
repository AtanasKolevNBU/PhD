from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import (StandardScaler, RobustScaler, MinMaxScaler,
                                   MaxAbsScaler, QuantileTransformer, PowerTransformer)
from scipy.signal import butter, filtfilt
import joblib

# ---------- stateless ops ----------
def _detrend_linear(y):
    t = np.arange(len(y))
    coef = np.polyfit(t, y, 1)
    return y - np.polyval(coef, t)

def _diff(y, d=1):
    z = y
    for _ in range(int(d)):
        z = np.diff(z, n=1)
    return z

# add your other stateless ops here (savgol, highpass, etc.)
def apply_stateless(y, name, params, fs):
    if name == "detrend_linear":
        return _detrend_linear(y)
    if name == "diff":
        return _diff(y, d=params.get("d", 1))
    elif name == "highpass":
        # Butterworth high-pass filter
        fc = float(params.get("fc", 1.0))             # cutoff frequency [Hz]
        order = int(params.get("order", 4))
        nyquist = 0.5 * fs
        if fc >= nyquist:
            fc = nyquist * 0.99                      # keep it below Nyquist
        b, a = butter(order, fc / nyquist, btype="highpass")
        return filtfilt(b, a, y)

    elif name == "savgol":
        from scipy.signal import savgol_filter
        win = int(params.get("win", 31)) | 1
        poly = int(params.get("poly", 2))
        return savgol_filter(y, win, poly)
    # ... your other stateless ops ...
    raise ValueError(f"Unknown stateless step: {name}")

# ---------- sklearn factory ----------
def make_transformer(name: str, params: Dict[str, Any], n_samples: int) -> Optional[Any]:
    if name == "standard_scaler":
        return StandardScaler(with_mean=params.get("with_mean", True),
                              with_std=params.get("with_std", True))
    if name == "robust_scaler":
        return RobustScaler(with_centering=params.get("with_centering", True),
                            with_scaling=params.get("with_scaling", True),
                            quantile_range=params.get("quantile_range", (25.0, 75.0)))
    if name == "minmax":
        return MinMaxScaler(feature_range=params.get("feature_range", (0.0, 1.0)))
    if name == "maxabs":
        return MaxAbsScaler()
    if name == "quantile":
        subsample = int(params.get("subsample", 100_000))
        nq_frac   = float(params.get("nq_frac", 1.0))
        # clamp n_quantiles safely (must be <= subsample and <= n_samples)
        n_q = max(10, min(int(n_samples * nq_frac), subsample, n_samples))
        return QuantileTransformer(n_quantiles=n_q,
                                   output_distribution=params.get("output_distribution", "normal"),
                                   subsample=subsample,
                                   random_state=params.get("random_state", 0),
                                   copy=True)
    if name == "power_transform":
        method = params.get("method", "yeo-johnson")  # 'yeo-johnson' | 'box-cox'
        # Box-Cox requires positivity; we will shift at transform-time if needed.
        return PowerTransformer(method=method, standardize=params.get("standardize", True))
    return None  # not a sklearn step

def is_sklearn_step(name: str) -> bool:
    return name in {"standard_scaler","robust_scaler","minmax","maxabs","quantile","power_transform"}

# ---------- fitted step container ----------
@dataclass
class FittedStep:
    name: str
    params: Dict[str, Any]
    transformer: Optional[Any]  # sklearn transformer or None for stateless

# ---------- fit the pipeline (global fit across training signals) ----------
def fit_pipeline(steps: List[Dict[str, Any]], train_signals: List[np.ndarray], fs: float) -> List[FittedStep]:
    """
    Fits trainable steps in order. For each step:
      - if stateless: apply it to all current train signals (updates them)
      - if sklearn:  fit on the concatenation of current train signals (after previous steps), then transform them (updates)
    Returns the list of FittedStep with fitted transformers kept for later use.
    """
    # make working copies
    currents = [np.asarray(s, dtype=float).ravel() for s in train_signals]
    fitted: List[FittedStep] = []

    for step in steps:
        name, params = step["name"], step.get("params", {})
        if is_sklearn_step(name):
            # compute current total sample count (after previous steps)
            n_samples = int(sum(len(s) for s in currents))
            tr = make_transformer(name, params, n_samples)
            # prepare training matrix for this step: concat all currents as one column
            X = np.concatenate([s.reshape(-1,1) for s in currents], axis=0)
            # special case: box-cox positivity
            if isinstance(tr, PowerTransformer) and tr.method == "box-cox":
                if np.min(X) <= 0:
                    X = X - np.min(X) + np.finfo(float).eps
            tr.fit(X)
            # transform each signal to feed the next step
            new_currents = []
            for s in currents:
                Z = s.reshape(-1,1)
                if isinstance(tr, PowerTransformer) and tr.method == "box-cox" and np.min(Z) <= 0:
                    Z = Z - np.min(Z) + np.finfo(float).eps
                s2 = tr.transform(Z).ravel()
                new_currents.append(s2)
            currents = new_currents
            fitted.append(FittedStep(name=name, params=params, transformer=tr))
        else:
            # stateless op: apply and propagate
            currents = [apply_stateless(s, name, params, fs) for s in currents]
            fitted.append(FittedStep(name=name, params=params, transformer=None))

    return fitted

# ---------- apply to a new signal ----------
def transform_with_fitted(fitted_steps: List[FittedStep], signal: np.ndarray, fs: float) -> np.ndarray:
    y = np.asarray(signal, dtype=float).ravel()
    for st in fitted_steps:
        if st.transformer is None:
            y = apply_stateless(y, st.name, st.params, fs)
        else:
            Z = y.reshape(-1,1)
            # keep Box-Cox positive
            if isinstance(st.transformer, PowerTransformer) and st.transformer.method == "box-cox" and np.min(Z) <= 0:
                Z = Z - np.min(Z) + np.finfo(float).eps
            y = st.transformer.transform(Z).ravel()
    return y

# ---------- batch apply ----------
def transform_batch(fitted_steps: List[FittedStep], signals: List[np.ndarray], fs: float) -> List[np.ndarray]:
    return [transform_with_fitted(fitted_steps, s, fs) for s in signals]

# ---------- persist / load ----------
def save_fitted_pipeline(fitted_steps: List[FittedStep], path: str):
    joblib.dump(fitted_steps, path)

def load_fitted_pipeline(path: str) -> List[FittedStep]:
    return joblib.load(path)