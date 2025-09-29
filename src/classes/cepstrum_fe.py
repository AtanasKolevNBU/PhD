import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from scipy.signal import get_window
from numpy.fft import rfft, irfft, fft, ifft
from typing import Literal, Tuple, Optional

@dataclass
class CepstrumFeatures:
    # Per-frame features (vectors with length = n_frames)
    dom_quef_ms: np.ndarray                # Dominant quefrency (ms)
    dom_freq_hz: np.ndarray                # Dominant frequency (Hz) inferred from dom quef
    cpp_db: np.ndarray                     # Cepstral Peak Prominence (dB)
    low_q_energy_db: np.ndarray            # Energy below lifter_queff_max (dB)
    # Optional summaries across frames
    dom_quef_ms_mean: float
    dom_quef_ms_std: float
    dom_freq_hz_mean: float
    dom_freq_hz_std: float
    cpp_db_mean: float
    cpp_db_std: float
    # Liftered cepstral coefficients (shape: n_frames x n_ceps)
    cceps: np.ndarray
    # Indices of rahmonic peaks (list of arrays per frame)
    rahmonic_idx: List[np.ndarray]

def _frame_signal(x: np.ndarray, fs: float, frame_len: float, hop_len: float, window: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Frame the signal x into overlapping frames and apply window.
    Returns (frames, win) with shape frames: (n_frames, n_samples_frame)
    """
    N = len(x)
    n_win = int(round(frame_len * fs))
    n_hop = int(round(hop_len * fs))
    if n_win <= 1:
        raise ValueError("frame_len too small.")
    if n_hop < 1:
        raise ValueError("hop_len too small.")
    n_frames = max(1, 1 + (N - n_win) // n_hop) if N >= n_win else 1

    frames = np.zeros((n_frames, n_win), dtype=float)
    win = get_window(window, n_win, fftbins=True).astype(float)
    if N < n_win:
        # zero-pad single frame
        frames[0, :N] = x
        frames[0] *= win
        return frames, win

    idx = 0
    for i in range(n_frames):
        seg = x[idx:idx+n_win]
        frames[i, :] = seg * win
        idx += n_hop
    return frames, win

def _real_cepstrum(frame: np.ndarray) -> np.ndarray:
    """
    Real cepstrum: IFFT{ log(|FFT| + eps) } using rFFT for real input.
    Output length equals frame length.
    """
    eps = 1e-12
    spec = rfft(frame)
    log_mag = np.log(np.abs(spec) + eps)
    ceps = irfft(log_mag, n=len(frame))
    return ceps

def _power_cepstrum(ceps: np.ndarray) -> np.ndarray:
    """
    Power cepstrum = ceps^2 (same length as ceps).
    """
    return ceps ** 2

def _complex_cepstrum(frame: np.ndarray) -> np.ndarray:
    """
    Complex cepstrum (optional): IFFT{ log(FFT{x}) } with phase unwrapping.
    Returns real sequence (the complex cepstrum is real-valued for real signals
    but involves handling the phase).
    """
    eps = 1e-12
    X = fft(frame)
    mag = np.abs(X) + eps
    phase = np.unwrap(np.angle(X))
    logX = np.log(mag) + 1j * phase
    c = np.real(ifft(logX))
    return c

def _find_dominant_quefrency(power_ceps: np.ndarray,
                             fs: float,
                             q_min_ms: float,
                             q_max_ms: float) -> Tuple[int, float]:
    """
    Find the dominant peak index in a restricted quefrency window (ms).
    Returns (peak_index, cpp_db) where cpp is prominence relative to local baseline.
    """
    n = len(power_ceps)
    q_axis_s = np.arange(n) / fs  # quefrency in seconds
    q_min = q_min_ms / 1000.0
    q_max = q_max_ms / 1000.0
    mask = (q_axis_s >= q_min) & (q_axis_s <= q_max)
    if not np.any(mask):
        return 0, 0.0
    region = power_ceps[mask]
    peak_rel_idx = np.argmax(region)
    peak_idx = np.where(mask)[0][0] + peak_rel_idx

    # A simple (robust) local baseline: median in the search window
    baseline = np.median(region)
    peak_val = power_ceps[peak_idx]
    # Convert to dB with small epsilon
    eps = 1e-20
    cpp_db = 10.0 * np.log10((peak_val + eps) / (baseline + eps))
    return peak_idx, cpp_db

def _rahmonic_indices(power_ceps: np.ndarray,
                      peak_idx: int,
                      max_mult: int = 6,
                      tolerance: int = 2) -> np.ndarray:
    """
    Return indices of rahmonic peaks near integer multiples of the dominant peak index.
    'tolerance' allows small index offsets due to discretization.
    """
    if peak_idx <= 0:
        return np.array([], dtype=int)
    idxs = []
    n = len(power_ceps)
    for m in range(2, max_mult + 1):
        target = m * peak_idx
        cand_min = max(0, target - tolerance)
        cand_max = min(n - 1, target + tolerance)
        local = power_ceps[cand_min:cand_max + 1]
        if local.size == 0:
            continue
        local_idx = np.argmax(local) + cand_min
        idxs.append(local_idx)
    return np.array(idxs, dtype=int)

def _low_quef_energy_db(power_ceps: np.ndarray, fs: float, q_max_ms: float) -> float:
    """
    Sum energy up to q_max_ms (excluding the DC term at 0 to avoid window energy bias).
    """
    q_max = int(round((q_max_ms / 1000.0) * fs))
    q_max = np.clip(q_max, 1, len(power_ceps) - 1)
    eps = 1e-20
    e = np.sum(power_ceps[1:q_max])
    return 10.0 * np.log10(e + eps)

def _lifter(ceps: np.ndarray, n_ceps: int) -> np.ndarray:
    """
    Keep first n_ceps coefficients (including c0). This is a simple low-time lifter.
    """
    n = len(ceps)
    n_keep = min(n_ceps, n)
    out = np.zeros(n, dtype=float)
    out[:n_keep] = ceps[:n_keep]
    return out[:n_keep]

class CepstrumFeatureExtractor:
    """
    Cepstrum feature extraction for vibration diagnostics.
    Works framewise and aggregates per-frame statistics.
    """

    def __init__(self,
                 fs: float,
                 frame_len: float = 0.25,     # seconds
                 hop_len: float = 0.125,      # seconds
                 window: str = "hann",
                 quef_search_ms: Tuple[float, float] = (0.3, 20.0),  # suitable for 50–3333 Hz periodicities
                 lifter_quef_ms: float = 3.0, # "low-quef" band upper bound for energy & noise floor
                 n_ceps: int = 20,
                 use_complex: bool = False):
        self.fs = fs
        self.frame_len = frame_len
        self.hop_len = hop_len
        self.window = window
        self.qmin_ms, self.qmax_ms = quef_search_ms
        self.lifter_quef_ms = lifter_quef_ms
        self.n_ceps = n_ceps
        self.use_complex = use_complex

    def transform(self, x: np.ndarray) -> CepstrumFeatures:
        """
        Compute framewise cepstra and derive features.
        """
        x = np.asarray(x, dtype=float).ravel()
        frames, _ = _frame_signal(x, self.fs, self.frame_len, self.hop_len, self.window)
        n_frames, n_win = frames.shape

        dom_quef_ms = np.zeros(n_frames, dtype=float)
        dom_freq_hz = np.zeros(n_frames, dtype=float)
        cpp_db = np.zeros(n_frames, dtype=float)
        low_q_energy_db = np.zeros(n_frames, dtype=float)
        cceps = np.zeros((n_frames, self.n_ceps), dtype=float)
        rahmonics: List[np.ndarray] = []

        for i in range(n_frames):
            f = frames[i]

            # Choose real or complex cepstrum
            if self.use_complex:
                ceps = _complex_cepstrum(f)
            else:
                ceps = _real_cepstrum(f)

            pceps = _power_cepstrum(ceps)

            # Dominant quefrency & CPP
            peak_idx, cpp = _find_dominant_quefrency(
                pceps, self.fs, self.qmin_ms, self.qmax_ms
            )
            q_s = peak_idx / self.fs if peak_idx > 0 else 0.0
            f0 = (1.0 / q_s) if q_s > 0 else 0.0

            dom_quef_ms[i] = 1000.0 * q_s
            dom_freq_hz[i] = f0
            cpp_db[i] = cpp

            # Low-quefrency energy (proxy for envelope/slow trends)
            low_q_energy_db[i] = _low_quef_energy_db(pceps, self.fs, self.lifter_quef_ms)

            # Liftered cepstral coefficients
            cceps[i, :] = _lifter(ceps, self.n_ceps)

            # Rahmonics
            rahmonics.append(_rahmonic_indices(pceps, peak_idx))

        feats = CepstrumFeatures(
            dom_quef_ms=dom_quef_ms,
            dom_freq_hz=dom_freq_hz,
            cpp_db=cpp_db,
            low_q_energy_db=low_q_energy_db,
            dom_quef_ms_mean=float(np.mean(dom_quef_ms)) if n_frames else 0.0,
            dom_quef_ms_std=float(np.std(dom_quef_ms)) if n_frames else 0.0,
            dom_freq_hz_mean=float(np.mean(dom_freq_hz)) if n_frames else 0.0,
            dom_freq_hz_std=float(np.std(dom_freq_hz)) if n_frames else 0.0,
            cpp_db_mean=float(np.mean(cpp_db)) if n_frames else 0.0,
            cpp_db_std=float(np.std(cpp_db)) if n_frames else 0.0,
            cceps=cceps,
            rahmonic_idx=rahmonics
        )
        return feats

    def to_feature_vector(self, feats: CepstrumFeatures) -> np.ndarray:
        """
        Flatten a compact set of statistics to a 1D feature vector suitable for ML.
        (You can customize what you include.)
        """
        stats = [
            feats.dom_quef_ms_mean, feats.dom_quef_ms_std,
            feats.dom_freq_hz_mean, feats.dom_freq_hz_std,
            feats.cpp_db_mean, feats.cpp_db_std,
            np.mean(feats.low_q_energy_db), np.std(feats.low_q_energy_db)
        ]
        # Aggregate cepstral coefficients (mean over frames)
        cceps_mean = np.mean(feats.cceps, axis=0) if feats.cceps.size else np.zeros(self.n_ceps)
        return np.concatenate([np.array(stats, dtype=float), cceps_mean])
    
    def _make_slices(self,
                 x: np.ndarray,
                 fs: float,
                 slice_len_s: float,
                 hop_s: float,
                 pad: Literal["drop", "zero", "keep_last"] = "drop") -> Tuple[np.ndarray, np.ndarray]:
        """
        Create overlapping slices of x. Returns (starts, stops) in sample indices.
        pad="drop":    drop tail shorter than slice_len
        pad="zero":    include last slice zero-padded to full length
        pad="keep_last": include the last shorter slice without padding
        """
        N = len(x)
        n_slice = int(round(slice_len_s * fs))
        n_hop = int(round(hop_s * fs))
        if n_slice <= 0 or n_hop <= 0:
            raise ValueError("slice_len_s and hop_s must be > 0.")

        starts = np.arange(0, max(N - n_slice + 1, 0), n_hop, dtype=int)
        stops = starts + n_slice

        if pad == "drop":
            mask = stops <= N
            starts, stops = starts[mask], stops[mask]
        elif pad == "zero":
            if len(starts) == 0 or stops[-1] < N:
                starts = np.append(starts, N - n_slice if N >= n_slice else 0)
                stops = np.append(stops, starts[-1] + n_slice)
        elif pad == "keep_last":
            if len(starts) == 0 or stops[-1] < N:
                starts = np.append(starts, starts[-1] + n_hop if len(starts) else 0)
                stops = np.append(stops, min(starts[-1] + n_slice, N))
        else:
            raise ValueError("pad must be 'drop', 'zero', or 'keep_last'.")

        return starts, stops

def batch_extract_cepstrum(
    x: np.ndarray,
    fs: float,
    slice_len_s: float = 2.0,
    hop_s: float = 1.0,
    frame_len_s: float = 0.25,
    frame_hop_s: float = 0.125,
    quef_search_ms: Tuple[float, float] = (0.2, 20.0),
    lifter_quef_ms: float = 5.0,
    n_ceps: int = 24,
    use_complex: bool = False,
    return_per_frame: bool = False
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Slices the signal and extracts a feature row per slice (compact vector).
    Optionally also returns per-frame diagnostics across all slices.

    Returns:
        df_slices: one row per slice (ML-ready)
        df_frames (optional): one row per internal frame with diagnostics
    """
    x = np.asarray(x, dtype=float).ravel()
    extractor = CepstrumFeatureExtractor(
        fs=fs,
        frame_len=frame_len_s,
        hop_len=frame_hop_s,
        window="hann",
        quef_search_ms=quef_search_ms,
        lifter_quef_ms=lifter_quef_ms,
        n_ceps=n_ceps,
        use_complex=use_complex
    )

    starts, stops = extractor._make_slices(x, fs, slice_len_s, hop_s, pad="zero")

    
    rows = []
    frame_rows = []

    for k, (s0, s1) in enumerate(zip(starts, stops)):
        seg = x[s0:s1]
        if len(seg) < int(round(frame_len_s * fs)):
            # zero-pad small edge case (should be rare due to pad="zero")
            seg = np.pad(seg, (0, int(round(frame_len_s * fs)) - len(seg)))

        feats = extractor.transform(seg)
        vec = extractor.to_feature_vector(feats)

        # --- per-slice compact row (ML-ready) ---
        row = {
            "slice_id": k,
            "start_s": s0 / fs,
            "end_s": s1 / fs,
            "duration_s": (s1 - s0) / fs,
            "dom_quef_ms_mean": feats.dom_quef_ms_mean,
            "dom_quef_ms_std":  feats.dom_quef_ms_std,
            "dom_freq_hz_mean": feats.dom_freq_hz_mean,
            "dom_freq_hz_std":  feats.dom_freq_hz_std,
            "cpp_db_mean":      feats.cpp_db_mean,
            "cpp_db_std":       feats.cpp_db_std,
            "low_q_energy_db_mean": float(np.mean(feats.low_q_energy_db)),
            "low_q_energy_db_std":  float(np.std(feats.low_q_energy_db))
        }
        # append liftered cepstral means (same as used in to_feature_vector)
        cceps_mean = np.mean(feats.cceps, axis=0) if feats.cceps.size else np.zeros(n_ceps)
        for i, v in enumerate(cceps_mean):
            row[f"cceps_mean_{i}"] = float(v)

        rows.append(row)

        # --- optional per-frame diagnostics ---
        if return_per_frame:
            n_frames = feats.cceps.shape[0]
            for i in range(n_frames):
                frame_rows.append({
                    "slice_id": k,
                    "frame_id": i,
                    "slice_start_s": s0 / fs,
                    "frame_mid_s": s0 / fs + (i * frame_hop_s) + frame_len_s / 2.0,
                    "dom_quef_ms": feats.dom_quef_ms[i],
                    "dom_freq_hz": feats.dom_freq_hz[i],
                    "cpp_db": feats.cpp_db[i],
                    "low_q_energy_db": feats.low_q_energy_db[i]
                })

    df_slices = pd.DataFrame(rows)
    df_frames = pd.DataFrame(frame_rows) if return_per_frame else None
    return df_slices, df_frames