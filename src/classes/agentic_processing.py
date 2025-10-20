import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import skew, boxcox_normmax, yeojohnson, boxcox, kurtosis
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import kpss
from scipy.signal import butter, filtfilt, savgol_filter
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler,
    QuantileTransformer, PowerTransformer
)
import optuna

def _as_col(x):
    x = np.asarray(x).reshape(-1, 1)
    return x

def _fit_apply_transformer(y, transformer):
    """Fits transformer on the current y (2D) and returns y_transformed (1D)."""
    Y = _as_col(y)
    yt = transformer.fit_transform(Y)
    yt = np.asarray(yt).ravel()
    # guard against NaN/Inf
    if not np.all(np.isfinite(yt)):
        return y  # fallback: return input unchanged
    # degenerate case: constant transform -> avoid zero-variance collapse unless desired
    if np.std(yt) == 0 and np.std(y) > 0:
        return y
    return yt

def _safe_n_quantiles(n_samples, nq_frac=1.0, subsample=100_000, min_q=10):
    # target based on fraction of samples
    target = int(max(min_q, np.floor(n_samples * float(nq_frac))))
    # sklearn also caps by subsample during fitting
    n_q = min(target, subsample, n_samples)
    return max(min_q, n_q)

def jb_penalty(x):
    from statsmodels.stats.stattools import jarque_bera
    _, p, _, _ = jarque_bera(x)
    return 0.0 if p > 0.05 else (0.05 - p)

def kpss_penalty(x):
    try:
        stat, p, _, _ = kpss(x, regression='c', nlags='auto')
    except Exception:
        return 1.0
    # KPSS null = stationarity; small p => non-stationary
    return 0.0 if p > 0.05 else (0.05 - p)

def lb_penalty(x, lags=(10,)):
    pvals = acorr_ljungbox(x, lags=lags, return_df=True)['lb_pvalue'].values
    return np.sum(np.clip(0.05 - pvals, 0, None))

def total_variation(x):
    return np.mean(np.abs(np.diff(x)))


# ---------- Bimodality metrics ----------
def ashman_D(mu1, mu2, s1, s2):
    return np.sqrt(2.0) * np.abs(mu1 - mu2) / np.sqrt(s1**2 + s2**2)

def gmm_bimodality(x, random_state=0):
    """Returns dict with delta_bic (>0 favors 2-GMM), Ashman's D, weights, and a 'strength' score."""
    x = np.asarray(x).reshape(-1, 1)
    g1 = GaussianMixture(n_components=1, covariance_type='full', random_state=random_state).fit(x)
    g2 = GaussianMixture(n_components=2, covariance_type='full', random_state=random_state).fit(x)
    bic1, bic2 = g1.bic(x), g2.bic(x)
    delta_bic = bic1 - bic2

    mu = g2.means_.ravel()
    s = np.sqrt(np.array([g2.covariances_[k].ravel()[0] for k in range(2)]))
    w = g2.weights_
    D = ashman_D(mu[0], mu[1], s[0], s[1])

    # A continuous "strength" in [0, +∞): positive means strong evidence for bimodality
    strength = max(0.0, delta_bic) * max(0.0, (D - 2.0))  # kicks in when D>2
    return {"delta_bic": float(delta_bic), "D": float(D), "weights": w, "strength": float(strength)}

def bimodality_coefficient(x):
    x = np.asarray(x)
    g1 = skew(x, bias=False)
    k  = kurtosis(x, fisher=False, bias=False)  # Pearson kurtosis
    if k <= 0:
        return 0.0
    return (g1**2 + 1.0) / k  # > ~0.555 suggests bimodality/heavy tails

# ---------- Configurable term ----------
class BimodalityTerm:
    """
    mode: 'discourage' | 'encourage' | 'ignore'
    method: 'gmm' (robust, default) | 'bc' (fast heuristic)
    weight: scaling factor for the term (>=0)
    thresholds: dict with optional keys for tailoring sensitivity
    """
    def __init__(self, mode="discourage", method="gmm", weight=1.0, thresholds=None, random_state=0):
        self.mode = mode
        self.method = method
        self.weight = float(weight)
        self.th = thresholds or {}
        self.rs = random_state

    def __call__(self, x):
        if self.mode == "ignore" or self.weight <= 0:
            return 0.0

        if self.method == "gmm":
            m = gmm_bimodality(x, random_state=self.rs)
            # Optional guards to avoid degenerate tiny components
            min_w = self.th.get("min_weight", 0.05)
            well_separated = (m["D"] > self.th.get("min_D", 2.0)) and np.all(m["weights"] >= min_w)
            strength = m["strength"] if well_separated else 0.0
            # Encourage => subtract from loss; Discourage => add to loss
            return (-self.weight * strength) if self.mode == "encourage" else (self.weight * strength)

        elif self.method == "bc":
            bc = bimodality_coefficient(x)
            # distance above/below the reference ~0.555
            ref = self.th.get("bc_ref", 0.555)
            margin = max(0.0, bc - ref)  # only considers "bimodal direction"
            return (-self.weight * margin) if self.mode == "encourage" else (self.weight * margin)

        else:
            raise ValueError(f"Unknown method: {self.method}")


def butter_highpass(x, fs, fc, order=4):
    b, a = butter(order, fc/(0.5*fs), btype='highpass')
    return filtfilt(b, a, x)

def apply_pipeline(x, fs, cfg):
    y = x.copy()
    for step in cfg:
        t = step['name']
        p = step.get('params', {})
        if t == 'robust_scale':
            med, iqr = np.median(y), (np.percentile(y,75)-np.percentile(y,25))
            y = (y - med) / (iqr if iqr>0 else 1.0)
        elif t == 'zscore':
            y = (y - np.mean(y)) / (np.std(y) or 1.0)
        elif t == 'yeo_johnson':
            lam = p.get('lam', 0.0)
            y = yeojohnson(y, lmbda=lam)
        elif t == 'boxcox':
            y_shift = y - np.min(y) + 1e-6
            lam = p.get('lam', None)
            y = boxcox(y_shift, lmbda=lam if lam is not None else boxcox_normmax(y_shift))
        elif t == 'power':
            gamma = p.get('gamma', 0.5)
            y = np.sign(y) * (np.abs(y) ** gamma)
        elif t == 'diff':
            d = p.get('d',1)
            for _ in range(d):
                y = np.diff(y, n=1)
        elif t == 'detrend_linear':
            t_idx = np.arange(len(y))
            coef = np.polyfit(t_idx, y, 1)
            y = y - np.polyval(coef, t_idx)
        elif t == 'savgol':
            y = savgol_filter(y, p.get('win', 31) | 1, p.get('poly', 2))
        elif t == 'highpass':
            y = butter_highpass(y, fs, p.get('fc', 1.0), p.get('order', 4))
        elif t == 'standard_scaler':
            tr = StandardScaler(with_mean=p.get('with_mean', True),
                                with_std=p.get('with_std', True))
            y = _fit_apply_transformer(y, tr); continue
        elif t == 'robust_scaler':
            tr = RobustScaler(
                with_centering=p.get('with_centering', True),
                with_scaling=p.get('with_scaling', True),
                quantile_range=p.get('quantile_range', (25.0, 75.0))
            )
            y = _fit_apply_transformer(y, tr); continue
        elif t == 'minmax':
            tr = MinMaxScaler(feature_range=p.get('feature_range', (0.0, 1.0)))
            y = _fit_apply_transformer(y, tr); continue
        elif t == 'maxabs':
            tr = MaxAbsScaler()
            y = _fit_apply_transformer(y, tr); continue
        elif t == 'quantile':
            n_samples = len(y)
            subsample = int(p.get('subsample', 100_000))
            nq_frac  = float(p.get('nq_frac', 1.0))
            outdist  = p.get('output_distribution', 'normal')  # 'normal'|'uniform'

            n_q = _safe_n_quantiles(n_samples, nq_frac=nq_frac, subsample=subsample, min_q=10)

            tr = QuantileTransformer(
                n_quantiles=n_q,
                output_distribution=outdist,
                subsample=subsample,
                random_state=p.get('random_state', 0),
                copy=True
            )
            y = _fit_apply_transformer(y, tr)
            continue
        # add more as needed
        
    return y

def objective(trial, signals, fs):
    # sample a small pipeline
    steps = []
    L = trial.suggest_int("n_steps", 1, 3)
    catalog = ['robust_scale','zscore','yeo_johnson','boxcox','power','diff','detrend_linear','savgol','highpass', 'standard_scaler','robust_scaler','minmax','maxabs','quantile','power_transform']
    for i in range(L):
        name = trial.suggest_categorical(f"step{i}", catalog)
        params = {}
        if name == 'yeo_johnson':
            params['lam'] = trial.suggest_float(f"yj_lam{i}", -2, 2)
        if name == 'boxcox':
            params['lam'] = trial.suggest_float(f"bc_lam{i}", -2, 2)
        if name == 'power':
            params['gamma'] = trial.suggest_float(f"pow_gam{i}", 0.2, 1.5)
        if name == 'diff':
            params['d'] = trial.suggest_int(f"diff_d{i}", 0, 2)
        if name == 'savgol':
            params['win'] = trial.suggest_int(f"sg_win{i}", 11, 151, step=2)
            params['poly'] = trial.suggest_int(f"sg_poly{i}", 2, 5)
        if name == 'highpass':
            params['fc'] = trial.suggest_float(f"hp_fc{i}", 0.2, fs*0.4)
            params['order'] = trial.suggest_int(f"hp_ord{i}", 2, 6)
        if name == 'robust_scaler':
            q_low  = trial.suggest_float(f"rs_qlo{i}", 5.0, 30.0)
            q_high = trial.suggest_float(f"rs_qhi{i}", 70.0, 95.0)
            if q_high <= q_low: q_high = q_low + 5.0
            params['quantile_range'] = (q_low, q_high)
            params['with_centering'] = trial.suggest_categorical(f"rs_ctr{i}", [True, False])
            params['with_scaling']   = trial.suggest_categorical(f"rs_scl{i}", [True, False])
        elif name == 'minmax':
            params['feature_range'] = trial.suggest_categorical(f"mm_range{i}", [(0.0,1.0), (-1.0,1.0)])
        elif name == 'quantile':
            params['output_distribution'] = trial.suggest_categorical(f"qt_out{i}", ['normal','uniform'])
            params['nq_frac'] = trial.suggest_float(f"qt_frac{i}", 0.2, 1.0)
            params['subsample'] = trial.suggest_int(f"qt_sub{i}", 10_000, 200_000, step=10_000)
            params['random_state'] = 0  # keep deterministic inside trials
        elif name == 'power_transform':
            params['method'] = trial.suggest_categorical(f"pt_meth{i}", ['yeo-johnson','box-cox'])
            params['standardize'] = trial.suggest_categorical(f"pt_std{i}", [True, False])
        steps.append({'name': name, 'params': params})

    # evaluate across N signals
    w = dict(skew=1.0, jb=0.5, stat=1.0, lb=0.5, tv=0.2, var=0.2)
    bi_term = BimodalityTerm(
    mode="discourage",           # or "encourage" or "ignore"
    method="gmm",                # or "bc"
    weight=0.8,                  # tune
    thresholds={"min_D": 2.0, "min_weight": 0.05, "bc_ref": 0.555},
    random_state=42
)
    trial.set_user_attr("steps_config", steps)  # store pipeline for later reuse
    losses = []
    for x in signals:
        x0 = np.asarray(x).ravel()
        y = apply_pipeline(x0, fs, steps)
        if y.size < 32:  # guard against excessive differencing
            losses.append(1e3); continue
        L = (
            w['skew']*abs(skew(y, bias=False))
            + w['jb']*jb_penalty(y)
            + w['stat']*kpss_penalty(y)
            + w['lb']*lb_penalty(y, lags=(10,20))
            + w['tv']*total_variation(y)
            + w['var']*abs(np.log((np.var(y)+1e-12)/(np.var(x0)+1e-12)))
            + bi_term(y)
        )
        losses.append(L)
    return float(np.mean(losses))

if __name__ == '__main__':
# usage:
    signals = [np.array([1,2,3]), np.array([1,2,3])]
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, signals=signals, fs=12000), n_trials=300)
    best_cfg = study.best_params  # -> reconstruct steps from params
