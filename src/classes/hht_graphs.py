import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

# ---- optional SciPy acceleration (auto-detected) ----
try:
    from scipy.interpolate import CubicSpline
    _HAVE_SCIPY_SPLINE = True
except Exception:
    _HAVE_SCIPY_SPLINE = False

try:
    from scipy.signal import hilbert as _scipy_hilbert
    _HAVE_SCIPY_HILBERT = True
except Exception:
    _HAVE_SCIPY_HILBERT = False


def _hilbert_analytic(x: np.ndarray) -> np.ndarray:
    """
    Analytic signal via Hilbert transform.
    Uses SciPy if available; otherwise FFT-based construction.
    x: (..., N)
    returns: complex analytic signal with same shape
    """
    if _HAVE_SCIPY_HILBERT:
        return _scipy_hilbert(x, axis=-1)

    # FFT-based Hilbert transform (last axis)
    Xf = np.fft.fft(x, axis=-1)
    N = x.shape[-1]
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = 1
        h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    return np.fft.ifft(Xf * h, axis=-1)


def _find_extrema(h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return indices of local maxima and minima for a 1D signal h.
    """
    dh = np.diff(h)
    # avoid plateaus: treat zeros by nudging with tiny noise-free rule
    # compute sign of derivative; treat zeros by carrying nearest nonzero sign
    sign = np.sign(dh)
    # propagate last nonzero sign across zeros
    nz = np.nonzero(sign)[0]
    if len(nz) > 0:
        last = nz[0]
        for i in range(len(sign)):
            if sign[i] == 0:
                sign[i] = sign[last]
            else:
                last = i
    # extrema where derivative changes sign
    max_idx = np.where((sign[:-1] > 0) & (sign[1:] < 0))[0] + 1
    min_idx = np.where((sign[:-1] < 0) & (sign[1:] > 0))[0] + 1
    return max_idx, min_idx


def _zero_crossings(h: np.ndarray) -> int:
    s = np.sign(h)
    # treat zeros as small numbers to stabilize counting
    s[s == 0] = 1
    return np.sum(s[:-1] * s[1:] < 0)

# =========================
# 1) IMFs + residual (overlay)
# =========================
def plot_imfs_compare(x1, t1, imfs1, res1,
                      x2, t2, imfs2, res2,
                      labels=("A","B"), title="IMFs comparison (overlay)"):
    K = max(len(imfs1), len(imfs2))
    rows = K + 1 + int(res1 is not None or res2 is not None)

    fig, axs = plt.subplots(rows, 1, figsize=(11, 1.6*rows), sharex=False)
    axs = np.atleast_1d(axs)

    axs[0].plot(t1, x1, label=labels[0], alpha=0.8, lw=1.0, color="C0")
    axs[0].plot(t2, x2, label=labels[1], alpha=0.8, lw=1.0, color="C1")
    axs[0].set_ylabel("x(t)")
    axs[0].set_title(title)
    axs[0].legend(loc="upper right")

    for i in range(K):
        ax = axs[i+1]
        if i < len(imfs1):
            ax.plot(t1, imfs1[i], alpha=0.9, lw=0.9, color="C0")
        if i < len(imfs2):
            ax.plot(t2, imfs2[i], alpha=0.9, lw=0.9, color="C1")
        ax.set_ylabel(f"IMF {i+1}")

    if res1 is not None or res2 is not None:
        ax = axs[-1]
        if res1 is not None:
            ax.plot(t1, res1, alpha=0.9, lw=0.9, color="C0")
        if res2 is not None:
            ax.plot(t2, res2, alpha=0.9, lw=0.9, color="C1")
        ax.set_ylabel("Остатък")
        ax.set_xlabel("Време, s")

    fig.tight_layout()
    plt.show()


# =====================================
# 2) Instantaneous frequency (per IMF)
# =====================================
def plot_instfreq_compare(t1, freq1, amp1,
                          t2, freq2, amp2,
                          fmax=None, labels=("A","B"),
                          title="Instantaneous frequency (compare)"):
    K = max(freq1.shape[0] if freq1.size else 0,
            freq2.shape[0] if freq2.size else 0)
    fig, axs = plt.subplots(K, 1, figsize=(11, 1.4*K), sharex=False)
    axs = np.atleast_1d(axs)

    def _mask(freq, amp):
        if freq.size == 0:
            return []
        thr = np.percentile(amp, 20, axis=1)  # per-IMF threshold
        masks = []
        for k in range(freq.shape[0]):
            m = np.isfinite(freq[k])
            if fmax is not None:
                m &= (freq[k] >= 0) & (freq[k] <= fmax)
            m &= amp[k] >= thr[k]
            masks.append(m)
        return masks

    m1 = _mask(freq1, amp1) if freq1.size else []
    m2 = _mask(freq2, amp2) if freq2.size else []

    for k in range(K):
        ax = axs[k]
        if k < (freq1.shape[0] if freq1.size else 0):
            f = np.clip(freq1[k], 0, fmax) if fmax is not None else freq1[k]
            ax.plot(t1[m1[k]], f[m1[k]], lw=0.9, alpha=0.9, color="C0", label=(labels[0] if k==0 else None))
        if k < (freq2.shape[0] if freq2.size else 0):
            f = np.clip(freq2[k], 0, fmax) if fmax is not None else freq2[k]
            ax.plot(t2[m2[k]], f[m2[k]], lw=0.9, alpha=0.9, color="C1", label=(labels[1] if k==0 else None))
        ax.set_ylabel(f"IMF{k+1} f [Hz]")
        if k == 0:
            ax.legend(loc="upper right")

    axs[-1].set_xlabel("Време, s")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


# ======================================================
# 3) Hilbert spectrum (time–frequency) — side-by-side
# ======================================================
def plot_hilbert_scatter_compare(t1, freq1, amp1,
                                 t2, freq2, amp2,
                                 fmax=None, amin_percentile=30,
                                 labels=("A","B"),
                                 title="Hilbert spectrum (scatter, side-by-side)"):
    def _prep(t, f, a):
        if f.size == 0:
            return np.array([]), np.array([]), np.array([])
        T = np.tile(t, (f.shape[0], 1))
        F = f.copy()
        A = a.copy()
        M = np.isfinite(F) & (F >= 0)
        if fmax is not None:
            M &= (F <= fmax)
        T, F, A = T[M], F[M], A[M]
        if A.size:
            thr = np.percentile(A, amin_percentile)
            keep = A >= thr
            return T[keep], F[keep], A[keep]
        return T, F, A

    T1, F1, A1 = _prep(t1, freq1, amp1)
    T2, F2, A2 = _prep(t2, freq2, amp2)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    sc1 = axs[0].scatter(T1, F1, c=A1, s=1, cmap="viridis")
    axs[0].set_title(labels[0]); axs[0].set_xlabel("time [s]"); axs[0].set_ylabel("frequency [Hz]")

    sc2 = axs[1].scatter(T2, F2, c=A2, s=1, cmap="viridis")
    axs[1].set_title(labels[1]); axs[1].set_xlabel("time [s]")

    # use a single colorbar scale for both panels
    vmin = np.nanmin([A1.min() if A1.size else np.nan, A2.min() if A2.size else np.nan])
    vmax = np.nanmax([A1.max() if A1.size else np.nan, A2.max() if A2.size else np.nan])
    for sc in (sc1, sc2):
        sc.set_clim(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(sc2, ax=axs.ravel().tolist(), location = 'right')
    cbar.set_label("Амплитуда")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


# ==========================================================
# 4) Hilbert energy map (2D histogram) — side-by-side (+Δ)
# ==========================================================
def plot_hilbert_energy_map_compare(t1, freq1, amp1,
                                    t2, freq2, amp2,
                                    fmax=None, nbins_t=400, nbins_f=400, power=2.0,
                                    labels=("A","B"), show_diff=True,
                                    title="Hilbert energy spectrum (side-by-side)"):
    def _hist(t, f, a):
        if f.size == 0:
            return None, None, None
        T = np.tile(t, (f.shape[0], 1)).ravel()
        F = f.ravel()
        W = (a**power).ravel()
        M = np.isfinite(F) & (F >= 0)
        if fmax is not None:
            M &= (F <= fmax)
        T, F, W = T[M], F[M], W[M]
        H, t_edges, f_edges = np.histogram2d(T, F, bins=[nbins_t, nbins_f], weights=W)
        Tm = 0.5*(t_edges[:-1] + t_edges[1:])
        Fm = 0.5*(f_edges[:-1] + f_edges[1:])
        return H, Tm, Fm

    H1, Tm1, Fm1 = _hist(t1, freq1, amp1)
    H2, Tm2, Fm2 = _hist(t2, freq2, amp2)
    if H1 is None or H2 is None:
        print("Nothing to plot (check inputs).")
        return

    # Use common axes: resample to same grid if needed (simple trim/pad to min)
    nt = min(H1.shape[0], H2.shape[0]); nf = min(H1.shape[1], H2.shape[1])
    H1v = H1[:nt, :nf]; H2v = H2[:nt, :nf]
    Tm = Tm1[:nt]; Fm = Fm1[:nf]

    ncols = 3 if show_diff else 2
    fig, axs = plt.subplots(1, ncols, figsize=(15 if show_diff else 10, 4), sharey=True, sharex=True)
    axs = np.atleast_1d(axs)

    vmax = max(np.percentile(H1v, 99), np.percentile(H2v, 99))
    im0 = axs[0].pcolormesh(Tm, Fm, H1v.T, shading="auto", vmin=0, vmax=vmax)
    axs[0].set_title(labels[0]); axs[0].set_xlabel("time [s]"); axs[0].set_ylabel("frequency [Hz]")

    im1 = axs[1].pcolormesh(Tm, Fm, H2v.T, shading="auto", vmin=0, vmax=vmax)
    axs[1].set_title(labels[1]); axs[1].set_xlabel("time [s]")

    cbar = fig.colorbar(im1, ax=axs[:2])
    cbar.set_label(f"∑ amplitude^{power}")

    if show_diff:
        D = (H2v - H1v)
        vmax_d = np.max(np.abs(np.percentile(D, [1, 99])))
        im2 = axs[2].pcolormesh(Tm, Fm, D.T, shading="auto", vmin=-vmax_d, vmax=vmax_d, cmap="bwr")
        axs[2].set_title("difference (B - A)")
        axs[2].set_xlabel("time [s]")
        fig.colorbar(im2, ax=axs[2]).set_label("Δ energy")

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


# ==========================================
# 5) Marginal spectrum (time-integrated)
# ==========================================
def plot_marginal_spectrum_compare(freq1, amp1, t1,
                                   freq2, amp2, t2,
                                   fmax=None, nbins_f=800, power=2.0,
                                   labels=("A","B"),
                                   title="Marginal Hilbert spectrum"):
    def _marginal(f, a, t):
        if f.size == 0:
            return np.array([]), np.array([])
        dt = np.median(np.diff(t))
        F = f.ravel()
        W = (a**power).ravel() * dt
        M = np.isfinite(F) & (F >= 0)
        if fmax is not None:
            M &= (F <= fmax)
        F, W = F[M], W[M]
        H, edges = np.histogram(F, bins=nbins_f, range=(0, fmax if fmax else F.max()), weights=W)
        Fm = 0.5*(edges[:-1] + edges[1:])
        return Fm, H

    F1, H1 = _marginal(freq1, amp1, t1)
    F2, H2 = _marginal(freq2, amp2, t2)
    if F1.size == 0 or F2.size == 0:
        print("Nothing to plot (check inputs).")
        return

    F = F1 if F1.size <= F2.size else F2  # simple alignment
    if F.size != F1.size:  # trim for overlay
        H1 = np.interp(F, F1, H1)
    if F.size != F2.size:
        H2 = np.interp(F, F2, H2)

    plt.figure(figsize=(10, 4))
    plt.plot(F, H1, label=labels[0], lw=1.2, color="C0")
    plt.plot(F, H2, label=labels[1], lw=1.2, color="C1")
    plt.xlabel("Честота, Hz")
    plt.ylabel(f"∫ Амплитуда^{power} dt")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

class EMD:
    """
    Minimal Empirical Mode Decomposition with linear envelopes (NumPy-only).
    If SciPy is available, uses cubic splines for smoother envelopes.

    Parameters
    ----------
    max_imfs : int or None
        Maximum number of IMFs to extract (None -> until residual has <2 extrema).
    max_siftings : int
        Max sifting iterations per IMF.
    sd_thresh : float
        Stopping threshold on normalized squared difference (Huang's SD).
    envelope_bc : {'auto','linear','cubic'}
        Envelope interpolation mode. 'auto' -> 'cubic' if SciPy available else 'linear'.
    mean_tol_ratio : float
        Additional stop: mean-envelope magnitude relative to signal RMS.
    """
    def __init__(self,
                 max_imfs: Optional[int] = None,
                 max_siftings: int = 50,
                 sd_thresh: float = 0.2,
                 envelope_bc: str = 'auto',
                 mean_tol_ratio: float = 0.05):
        self.max_imfs = max_imfs
        self.max_siftings = max_siftings
        self.sd_thresh = sd_thresh
        self.mean_tol_ratio = mean_tol_ratio
        if envelope_bc == 'auto':
            self.envelope_mode = 'cubic' if _HAVE_SCIPY_SPLINE else 'linear'
        else:
            self.envelope_mode = envelope_bc

    def _interp_envelope(self, t: np.ndarray, h: np.ndarray, idx: np.ndarray) -> Optional[np.ndarray]:
        """
        Build upper/lower envelope through extrema indices.
        Linear by default; cubic if SciPy available and selected.
        Extends to endpoints by holding first/last extrema values.
        """
        n = len(h)
        if idx.size < 2:
            return None

        # ensure endpoints present (simple hold extension)
        if idx[0] != 0:
            idx = np.r_[0, idx]
            vals = np.r_[h[idx[1]], h[idx[1:]]]
        else:
            vals = h[idx]
        if idx[-1] != n - 1:
            idx = np.r_[idx, n - 1]
            vals = np.r_[vals, vals[-1]]

        # ensure strict monotonicity in idx for cubic
        uniq_idx, uniq_pos = np.unique(idx, return_index=True)
        vals = vals[uniq_pos]

        if self.envelope_mode == 'cubic' and _HAVE_SCIPY_SPLINE and uniq_idx.size >= 3:
            cs = CubicSpline(t[uniq_idx], vals, bc_type='natural', extrapolate=True)
            return cs(t)
        else:
            return np.interp(t, t[uniq_idx], vals)

    def _extract_one_imf(self, t: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Sift residual r to extract one IMF. Returns (imf, success_flag)
        """
        h = r.copy()
        for _ in range(self.max_siftings):
            max_idx, min_idx = _find_extrema(h)
            if max_idx.size < 2 or min_idx.size < 2:
                # cannot form envelopes — stop
                return h, False

            env_max = self._interp_envelope(t, h, max_idx)
            env_min = self._interp_envelope(t, h, min_idx)
            if env_max is None or env_min is None:
                return h, False
            m = 0.5 * (env_max + env_min)
            h_new = h - m

            # SD stopping criterion
            denom = np.sum(h ** 2) + 1e-15
            sd = np.sum((h - h_new) ** 2) / denom

            # IMF property checks
            nz = _zero_crossings(h_new)
            ne = _find_extrema(h_new)[0].size + _find_extrema(h_new)[1].size
            imf_like = abs(ne - nz) <= 1
            mean_small = (np.mean(np.abs(m)) <= self.mean_tol_ratio * (np.sqrt(denom / len(h))))

            h = h_new
            if sd < self.sd_thresh and imf_like and mean_small:
                return h, True

        # reached max sifts — accept best effort
        return h, True

    def decompose(self, x: np.ndarray, t: Optional[np.ndarray] = None) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Decompose x into IMFs and residual.
        """
        x = np.asarray(x).astype(float)
        n = len(x)
        if t is None:
            t = np.arange(n, dtype=float)

        imfs = []
        r = x.copy()
        imf_count = 0
        while True:
            max_idx, min_idx = _find_extrema(r)
            if (max_idx.size + min_idx.size) < 2:  # no meaningful oscillation left
                break
            if self.max_imfs is not None and imf_count >= self.max_imfs:
                break

            imf, ok = self._extract_one_imf(t, r)
            imfs.append(imf)
            r = r - imf
            imf_count += 1
            if not ok:
                break

        return imfs, r


class HHT(EMD):
    """
    HHT = EMD + Hilbert spectral analysis (instantaneous amplitude & frequency).
    """
    @staticmethod
    def hilbert_analysis(imfs: List[np.ndarray], fs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute analytic signal, amplitude, phase, and instantaneous frequency for each IMF.

        Returns
        -------
        amp : (K, N) instantaneous amplitude
        phase : (K, N) unwrapped phase [rad]
        freq : (K, N) instantaneous frequency [Hz]
        """
        if len(imfs) == 0:
            return (np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0)))
        X = np.stack(imfs, axis=0)  # (K, N)
        z = _hilbert_analytic(X)    # analytic signal
        amp = np.abs(z)
        phase = np.unwrap(np.angle(z), axis=-1)
        # derivative of phase -> angular frequency -> Hz
        dphi = np.gradient(phase, axis=-1)
        omega = dphi * fs
        freq = omega / (2 * np.pi)
        return amp, phase, freq

    def transform(self, x: np.ndarray, fs: float, t: Optional[np.ndarray] = None):
        """
        Convenience: do EMD then Hilbert analysis.

        Returns dict with keys:
        - 'imfs': list of IMFs
        - 'residual': residual signal
        - 'amplitude': (K, N)
        - 'phase': (K, N)
        - 'frequency': (K, N) in Hz
        """
        imfs, r = self.decompose(x, t)
        amp, phase, freq = self.hilbert_analysis(imfs, fs)
        return dict(imfs=imfs, residual=r, amplitude=amp, phase=phase, frequency=freq)
    

# ------------------------- Example usage -------------------------
if __name__ == "__main__":
    # Synthetic AM-FM test signal
    fs = 2000
    N = 1000
    t = np.arange(N) / fs

    # component 1: 8 Hz with slow AM
    c1 = (1.0 + 0.3 * np.sin(2 * np.pi * 0.2 * t)) * np.sin(2 * np.pi * 8.0 * t)
    # component 2: chirp-like component (frequency increases)
    c2 = 0.6 * np.sin(2 * np.pi * (2.0 * t + 0.001 * (t ** 2)))
    # component 3: higher frequency
    c3 = 0.3 * np.sin(2 * np.pi * 30.0 * t)
    x = c1 + c2 + c3 + 0.05 * np.random.randn(N)

    hht = HHT(max_imfs=6, max_siftings=100, sd_thresh=0.2, envelope_bc='auto')
    result = hht.transform(x, fs, t)

    imfs = result["imfs"]
    residual = result["residual"]
    amp = result["amplitude"]
    freq = result["frequency"]

    print(f"Extracted {len(imfs)} IMFs")
    for i, imf in enumerate(imfs, 1):
        print(f"IMF {i}: std={np.std(imf):.4f}, length={len(imf)}")
    print(f"Residual std: {np.std(residual):.4f}")

    # Example: average instantaneous frequency per IMF (ignoring edges)
    if freq.size:
        mean_freqs = np.nanmean(freq[:, 10:-10], axis=1)
        for i, mf in enumerate(mean_freqs, 1):
            print(f"Mean inst. freq of IMF {i}: {mf:.2f} Hz")