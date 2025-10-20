import numpy as np
from scipy.stats import skew, boxcox_normmax, yeojohnson, boxcox
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import kpss
from scipy.signal import butter, filtfilt, savgol_filter
import optuna

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
        # add more as needed
    return y

def objective(trial, signals, fs):
    # sample a small pipeline
    steps = []
    L = trial.suggest_int("n_steps", 1, 3)
    catalog = ['robust_scale','zscore','yeo_johnson','boxcox','power','diff','detrend_linear','savgol','highpass']
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
        steps.append({'name': name, 'params': params})

    # evaluate across N signals
    w = dict(skew=1.0, jb=0.5, stat=1.0, lb=0.5, tv=0.2, var=0.2)
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
        )
        losses.append(L)
    return float(np.mean(losses))

if __name__ == '__main__':
# usage:
    signals = [np.array([1,2,3]), np.array([1,2,3])]
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, signals=signals, fs=12000), n_trials=300)
    best_cfg = study.best_params  # -> reconstruct steps from params
