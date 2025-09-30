import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Iterable

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet, EmpiricalCovariance
from sklearn.cluster import KMeans

try:
    from scipy.stats import chi2
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


@dataclass
class PCAMahalanobisConfig:
    # PCA
    n_components: Optional[int] = None
    var_target: float = 0.95
    standardize: bool = True

    # Covariance
    robust: bool = True

    # Threshold labeling (used when cluster_mode='threshold')
    q_ok: float = 0.90
    q_warn: float = 0.99

    # Baseline identification
    baseline_col: str = "label"
    baseline_value: Union[str, int, bool] = "baseline"

    # Feature columns
    feature_cols: Optional[Iterable[str]] = None

    # Clustering mode
    cluster_mode: str = "threshold"       # 'threshold' or 'kmeans'
    kmeans_space: str = "scores"          # 'scores' or 'distance'
    random_state: Optional[int] = 0
    kmeans_n_init: int = 10


class PCAMahalanobisClassifier:
    """
    PCA -> baseline covariance -> Mahalanobis distances.
    Labeling:
      - cluster_mode='threshold': Baseline | OK/Warning/Error by chi2 cutoffs
      - cluster_mode='kmeans'   : Baseline | KMeans(3) on non-baseline
        (clusters ordered by mean distance -> OK < Warning < Error).
    """
    def __init__(self, config: PCAMahalanobisConfig = PCAMahalanobisConfig()):
        self.config = config
        self.scaler_: Optional[StandardScaler] = None
        self.pca_: Optional[PCA] = None
        self.cov_: Optional[Union[MinCovDet, EmpiricalCovariance]] = None
        self.mu_: Optional[np.ndarray] = None
        self.precision_: Optional[np.ndarray] = None
        self.k_: Optional[int] = None
        self.t_ok_: Optional[float] = None
        self.t_warn_: Optional[float] = None
        self.feature_cols_: Optional[list] = None

        # KMeans state
        self.kmeans_: Optional[KMeans] = None
        self.kmeans_label_map_: Optional[dict] = None  # raw_cluster_label -> {'OK','Warning','Error'}

    def _select_feature_cols(self, df: pd.DataFrame) -> list:
        if self.config.feature_cols is not None:
            return list(self.config.feature_cols)
        exc = {self.config.baseline_col}
        return [c for c in df.columns if c not in exc and pd.api.types.is_numeric_dtype(df[c])]

    def _fit_scaler_pca(self, X: np.ndarray):
        if self.config.standardize:
            self.scaler_ = StandardScaler()
            Xs = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None
            Xs = X

        pca_probe = PCA(svd_solver='full', random_state=self.config.random_state)
        pca_probe.fit(Xs)
        if self.config.n_components is None:
            cum = np.cumsum(pca_probe.explained_variance_ratio_)
            k = int(np.searchsorted(cum, self.config.var_target) + 1)
            k = max(1, k)
        else:
            k = int(self.config.n_components)

        pca = PCA(n_components=k, svd_solver='full', random_state=self.config.random_state)
        Z = pca.fit_transform(Xs)
        self.pca_ = pca
        self.k_ = k
        return Z

    def _transform_to_scores(self, X: np.ndarray) -> np.ndarray:
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        return self.pca_.transform(X)

    def _fit_covariance_on_baseline(self, Z_baseline: np.ndarray) -> None:
        if self.config.robust:
            cov = MinCovDet(random_state=self.config.random_state).fit(Z_baseline)
        else:
            cov = EmpiricalCovariance().fit(Z_baseline)
        self.cov_ = cov
        self.mu_ = cov.location_
        self.precision_ = cov.precision_

    def _mahalanobis_d2(self, Z: np.ndarray) -> np.ndarray:
        D = Z - self.mu_
        d2 = np.einsum('ij,jk,ik->i', D, self.precision_, D)
        return np.clip(d2, 0.0, None)

    def _set_thresholds(self, d2_baseline: np.ndarray):
        if self.config.cluster_mode == "threshold":
            if _HAVE_SCIPY:
                self.t_ok_ = float(chi2.ppf(self.config.q_ok, df=self.k_))
                self.t_warn_ = float(chi2.ppf(self.config.q_warn, df=self.k_))
            else:
                self.t_ok_ = float(np.quantile(d2_baseline, self.config.q_ok))
                self.t_warn_ = float(np.quantile(d2_baseline, self.config.q_warn))

    def _fit_kmeans(self, Z_all: np.ndarray, d2_all: np.ndarray, baseline_mask: np.ndarray):
        # Prepare non-baseline training view
        nb = ~baseline_mask
        if nb.sum() < 3:
            # Too few samples; fall back to thresholding
            self.config.cluster_mode = "threshold"
            return

        if self.config.kmeans_space == "distance":
            Xkm = d2_all[nb].reshape(-1, 1)          # 1D on distance
        else:
            Xkm = Z_all[nb]                           # multi-D on PCA scores

        km = KMeans(
            n_clusters=3,
            n_init=self.config.kmeans_n_init,
            random_state=self.config.random_state
        ).fit(Xkm)
        self.kmeans_ = km

        # Map raw clusters -> OK/Warning/Error by mean MD^2 per cluster (ascending)
        cluster_ids = np.unique(km.labels_)
        means = []
        for cid in cluster_ids:
            idx = (km.labels_ == cid)
            means.append((cid, float(d2_all[nb][idx].mean())))
        # sort by mean d2
        means.sort(key=lambda t: t[1])
        ordered = [cid for cid, _ in means]  # low -> high
        label_map = {}
        for cid, name in zip(ordered, ["OK", "Warning", "Error"]):
            label_map[cid] = name
        self.kmeans_label_map_ = label_map

    def fit(self, df: pd.DataFrame):
        self.feature_cols_ = self._select_feature_cols(df)
        X = df[self.feature_cols_].to_numpy()
        baseline_mask = (df[self.config.baseline_col] == self.config.baseline_value).to_numpy()

        # PCA on all; covariance from baseline
        Z_all = self._fit_scaler_pca(X)
        Z_base = Z_all[baseline_mask]
        if Z_base.shape[0] < self.k_:
            raise ValueError(
                f"Not enough baseline samples ({Z_base.shape[0]}) for PCA dimension k={self.k_}. "
                "Reduce n_components or gather more baseline data."
            )
        self._fit_covariance_on_baseline(Z_base)

        # distances for all (used by both modes)
        d2_all = self._mahalanobis_d2(Z_all)
        self._set_thresholds(d2_baseline=d2_all[baseline_mask])

        # KMeans path (optional)
        if self.config.cluster_mode == "kmeans":
            self._fit_kmeans(Z_all, d2_all, baseline_mask)

        # Cache for transform/predict
        self._Z_all_fit_cache_ = Z_all
        self._d2_all_fit_cache_ = d2_all
        self._baseline_mask_fit_cache_ = baseline_mask
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_cols_].to_numpy()
        Z = self._transform_to_scores(X)
        d2 = self._mahalanobis_d2(Z)
        out = pd.DataFrame(Z, index=df.index, columns=[f"PC{i+1}" for i in range(self.k_)])
        out["md2"] = d2
        out["md"] = np.sqrt(d2)
        return out

    def predict(self, df: pd.DataFrame) -> pd.Series:
        X = df[self.feature_cols_].to_numpy()
        Z = self._transform_to_scores(X)
        d2 = self._mahalanobis_d2(Z)
        baseline_mask = (df[self.config.baseline_col] == self.config.baseline_value).to_numpy()

        labels = np.empty(Z.shape[0], dtype=object)
        labels[baseline_mask] = "Baseline"

        nb = ~baseline_mask
        if self.config.cluster_mode == "kmeans" and self.kmeans_ is not None and nb.sum() > 0:
            if self.config.kmeans_space == "distance":
                Xkm = d2[nb].reshape(-1, 1)
            else:
                Xkm = Z[nb]
            raw = self.kmeans_.predict(Xkm)
            mapped = [self.kmeans_label_map_.get(r, "Warning") for r in raw]
            labels[nb] = mapped
        else:
            # Threshold mode
            labels[nb & (d2 <= self.t_ok_)] = "OK"
            labels[nb & (d2 > self.t_ok_) & (d2 <= self.t_warn_)] = "Warning"
            labels[nb & (d2 > self.t_warn_)] = "Error"

        return pd.Series(labels, index=df.index, name="State")

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        res = self.transform(df)
        res["State"] = self.predict(df)
        return res

    # Convenience
    def thresholds(self) -> Tuple[Optional[float], Optional[float]]:
        return self.t_ok_, self.t_warn_

    def components_(self) -> pd.DataFrame:
        return pd.DataFrame(self.pca_.components_,
                            columns=self.feature_cols_,
                            index=[f"PC{i+1}" for i in range(self.k_)])



# ============== Example Usage =================
# Suppose the DataFrame is 'df' with:
# - numeric feature columns
# - a column 'label' that equals 'baseline' for baseline rows (change in config if needed)

if __name__ == '__main__':
    df = pd.DataFrame()
    cfg = PCAMahalanobisConfig(
        n_components=None,      # auto-select k to reach var_target
        var_target=0.95,
        standardize=True,
        robust=True,            # MinCovDet for baseline covariance
        q_ok=0.90,
        q_warn=0.99,
        baseline_col="label",
        baseline_value="baseline",
        feature_cols=None       # infer numeric columns except 'label'; or pass an explicit list
    )

    clf = PCAMahalanobisClassifier(cfg)
    result = clf.fit_predict(df)

    # 'result' includes PC scores, md2 (squared Mahalanobis), md, and 'State' in {"Baseline","OK","Warning","Error"}.
    # Example:
    print(clf.thresholds())     # (t_ok, t_warn) on chi-square d^2 scale
    print(result[["md2","State"]].head())

    # If you need the PCA loadings:
    loadings = clf.components_()