from hht_graphs import HHT
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import kurtosis, skew

try:
    from scipy.signal import hilbert as _hilbert
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

hht = HHT(max_imfs=6, max_siftings=100, sd_thresh=0.2, envelope_bc='auto')

class HHT_FeatureExtraction:

    def __init__(self):
        pass

    def _hilbert_analytic(self, x: np.ndarray) -> np.ndarray:
        if _HAVE_SCIPY:
            return _hilbert(x, axis=-1)
        # FFT-based fallback
        Xf = np.fft.fft(x, axis=-1)
        N = x.shape[-1]
        h = np.zeros(N)
        if N % 2 == 0:
            h[0] = h[N//2] = 1
            h[1:N//2] = 2
        else:
            h[0] = 1
            h[1:(N+1)//2] = 2
        return np.fft.ifft(Xf * h, axis=-1)

    def _safe(self, x): 
        x = np.asarray(x)
        return x[np.isfinite(x)]

    def _spec_at(self, f, F, A, bw_bins=1):
        """Amplitude around target freq f in FFT grid (median in ±bw_bins)."""
        if f <= 0 or f > F[-1]:
            return 0.0
        j = np.argmin(np.abs(F - f))
        j0 = max(0, j - bw_bins); j1 = min(len(F)-1, j + bw_bins)
        return float(np.median(A[j0:j1+1]))

    def hht_feature_vector(self, result: Dict, fs: float,
                       t: Optional[np.ndarray] = None,
                       edge_trim: float = 0.02,
                       fmax: Optional[float] = None,
                       band_edges: Optional[List[Tuple[float,float]]] = None,
                       fault_freqs: Optional[Dict[str, float]] = None,
                       resonant_select: str = "top_energy",
                       resonant_k: int = 2,
                       k_target: Optional[int] = None   # <--- NEW
                       ) -> Tuple[np.ndarray, List[str]]:
        """
        Build an HHT feature vector from:
        result['imfs'], result['amplitude'], result['frequency'], result['residual'].
        Returns (X, feature_names).
        """
        
        imfs: List[np.ndarray] = result["imfs"]
        amp  = result["amplitude"]      # (K, N)
        freq = result["frequency"]      # (K, N)
        K = len(imfs)
        if k_target is None:
            k_target = K  # default: keep current behavior

        if len(imfs) == 0:
            return np.zeros(1), ["empty_hht"]

        N = amp.shape[1]
        if t is None:
            t = np.arange(N)/fs
        dt = np.median(np.diff(t))
        fmax = fmax if fmax is not None else 0.45*fs

        # masks: trim edges + valid Hz
        i0 = int(edge_trim*N); i1 = int((1-edge_trim)*N)
        mask = np.zeros_like(freq, dtype=bool)
        mask[:, i0:i1] = True
        mask &= np.isfinite(freq) & (freq >= 0) & (freq <= fmax) & np.isfinite(amp)

        # ---------- Per-IMF features ----------
        feats = []
        names = []
        K = len(imfs)
        total_energy = 1e-12 + np.sum((amp**2)*mask, dtype=float) * dt

        # amplitude-weighted center frequency to help resonant selection
        aw_center = np.zeros(K)
        per_imf_energy = np.zeros(K)

        for k in range(K):
            xk = imfs[k]
            mk = mask[k]
            ak = amp[k, mk]; fk = freq[k, mk]

            # time-domain
            rms = np.sqrt(np.mean(xk**2))
            cf  = np.max(np.abs(xk)) / (rms + 1e-12)
            kur = kurtosis(self._safe(xk), fisher=False, bias=False)
            skw = skew(self._safe(xk), bias=False)
            zcr = np.mean(np.diff(np.signbit(xk)) != 0)

            feats += [rms, cf, kur, skw, zcr]
            names += [f"imf{k+1}_rms", f"imf{k+1}_crest", f"imf{k+1}_kurt",
                    f"imf{k+1}_skew", f"imf{k+1}_zcr"]

            # Hilbert amplitude/frequency stats
            a_mean = np.mean(ak) if ak.size else 0.0
            a_std  = np.std(ak)  if ak.size else 0.0
            a_p95  = np.percentile(ak, 95) if ak.size else 0.0

            # amplitude-weighted frequency stats
            if ak.size:
                w = ak
                f_mean = float(np.sum(fk*w) / (np.sum(w) + 1e-12))
                f_std  = float(np.sqrt(np.sum(((fk - f_mean)**2)*w) / (np.sum(w) + 1e-12)))
            else:
                f_mean, f_std = 0.0, 0.0

            feats += [a_mean, a_std, a_p95, f_mean, f_std]
            names += [f"imf{k+1}_amp_mean", f"imf{k+1}_amp_std", f"imf{k+1}_amp_p95",
                    f"imf{k+1}_f_mean", f"imf{k+1}_f_std"]

            # energy and proportion
            Ek = float(np.sum((amp[k]**2)*mk) * dt)
            per_imf_energy[k] = Ek
            aw_center[k] = f_mean
            feats += [Ek, Ek/(total_energy)]
            names += [f"imf{k+1}_Hilbert_energy", f"imf{k+1}_energy_frac"]
        if k_target > K:
            base_names = ["rms","crest","kurt","skew","zcr",
                        "amp_mean","amp_std","amp_p95","f_mean","f_std",
                        "Hilbert_energy","energy_frac"]
            for k in range(K, k_target):
                feats += [0.0]*12
                names += [f"imf{k+1}_{nm}" for nm in base_names]

        # ---------- Global Hilbert spectrum features ----------
        # Marginal spectrum via histogram
        F = freq[mask]
        W = (amp[mask]**2) * dt  # energy weights
        if F.size:
            nb = min(800, max(50, int(np.sqrt(F.size))))
            H, edges = np.histogram(F, bins=nb, range=(0, fmax), weights=W)
            Fm = 0.5*(edges[:-1] + edges[1:])
            Hn = H / (np.sum(H) + 1e-12)
            # moments
            centroid = float(np.sum(Fm*Hn))
            spread   = float(np.sqrt(np.sum(((Fm-centroid)**2)*Hn)))
            entropy  = float(-np.sum(Hn*np.log(Hn + 1e-12)))
            feats += [centroid, spread, entropy]
            names += ["hilbert_centroid", "hilbert_spread", "hilbert_entropy"]
            # optional band energies
            if band_edges:
                for i,(f1,f2) in enumerate(band_edges,1):
                    m = (F >= f1) & (F < f2)
                    feats.append(float(np.sum(W[m])/(total_energy)))
                    names.append(f"band_{i}_{int(f1)}-{int(f2)}Hz_energy_frac")
        else:
            feats += [0.0, 0.0, 0.0]
            names += ["hilbert_centroid", "hilbert_spread", "hilbert_entropy"]

        # ---------- Resonant-IMF envelope spectrum (fault-rate features) ----------
        # pick resonant IMFs
        if K:
            if resonant_select == "top_energy":
                sel = np.argsort(per_imf_energy)[::-1][:max(1, resonant_k)]
            elif resonant_select == "high_freq":
                sel = np.argsort(aw_center)[::-1][:max(1, resonant_k)]
            else:  # custom list of indices
                sel = np.atleast_1d(resonant_select).astype(int)
                sel = sel[(sel>=0) & (sel<K)]
            y = np.sum([imfs[i] for i in sel], axis=0)
            env = np.abs(self._hilbert_analytic(y))
            # FFT of envelope
            E = np.abs(np.fft.rfft(env))
            Fenv = np.fft.rfftfreq(len(env), d=dt)

            # summary peaks (top 5 below 0.5*fmax_env)
            mlim = int(0.5*len(Fenv))
            top_idx = np.argsort(E[:mlim])[::-1][:5]
            for j,idx in enumerate(top_idx,1):
                feats += [float(Fenv[idx]), float(E[idx])]
                names += [f"env_peak{j}_freq", f"env_peak{j}_amp"]

            # targeted fault frequencies (if provided)
            if fault_freqs:
                for key, f0 in fault_freqs.items():
                    a0 = self._spec_at(f0, Fenv, E, bw_bins=1)
                    a2 = self._spec_at(2*f0, Fenv, E, bw_bins=1)
                    # local SNR vs ±5 bins
                    j = np.argmin(np.abs(Fenv - f0))
                    j0 = max(0, j-5); j1 = min(len(E)-1, j+5)
                    noise = np.median(E[j0:j1+1])
                    feats += [a0, a2, a0/(noise+1e-12)]
                    names += [f"{key}_amp", f"{key}_2x_amp", f"{key}_snr"]
        return np.array(feats, dtype=float), names
    
    def windowed_hht_table(self, x, fs, win_s=0.2, step_s=0.05, k_target=6):
        L = int(round(win_s*fs)); S = int(round(step_s*fs))
        if L < 300:  # ~0.094 s at 3.2 kHz
            raise ValueError(f"win_s={win_s}s gives only {L} samples; increase win_s for fs={fs} Hz.")
        rows, names = [], None
        for i in range(0, len(x)-L+1, S):
            seg = x[i:i+L]
            res = hht.transform(seg, fs)
            feats, names = self.hht_feature_vector(res, fs, k_target=k_target, fmax=0.45*fs)
            rows.append(feats)
        return np.vstack(rows), names