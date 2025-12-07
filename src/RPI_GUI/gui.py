#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import threading
import queue
import time
import math
import random
from dataclasses import dataclass
from typing import List

import numpy as np
import tkinter as tk
from tkinter import ttk

import src.classes.cepstrum_fe
import src.functions.helpers
import src.classes.AD_Mahalanobis_Cluster as admc




# ============= CONFIG ==================

SAMPLE_RATE_HZ = 12000          # simulated sensor sample rate
WINDOW_SIZE = 512              # samples per diagnostic window
WINDOW_INTERVAL_S = 2        # how often we run diagnostics
FAULT_PROBABILITY = 0.02       # chance that a given window is "faulty"

# ============= DATA TYPES ==============

@dataclass
class SensorWindow:
    timestamp: float
    samples: np.ndarray
    is_fault_injected: bool  # hidden ground truth for testing


@dataclass
class DiagnosticResult:
    timestamp: float
    feat1: float
    feat2: float
    feat1_name: str
    feat2_name: str
    status: str            # "OK" / "WARNING" / "FAULT"
    reason: str
    feature_mode: str      # "time" / "cepstrum" / "hht"


# ============= HELPER: HILBERT (for HHT) ==============

def hilbert_numpy(x: np.ndarray) -> np.ndarray:
    """
    Simple Hilbert transform implementation using FFT.
    Returns the analytic signal.
    """
    N = len(x)
    Xf = np.fft.fft(x)
    h = np.zeros(N)

    if N % 2 == 0:
        h[0] = 1.0
        h[N // 2] = 1.0
        h[1:N // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(N + 1) // 2] = 2.0

    return np.fft.ifft(Xf * h)


# ============= SENSOR SIMULATOR ==============

class SensorSimulator(threading.Thread):
    """
    Simulates a vibration sensor: sine + noise + occasional impulses (faults).
    Generates windows and pushes them to a queue.
    """
    def __init__(self, out_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.t = 0.0

    def run(self):
        dt = 1.0 / SAMPLE_RATE_HZ
        samples_buffer: List[float] = []

        while not self.stop_event.is_set():
            # Base signal: sine wave + small noise
            base = math.sin(2 * math.pi * 50 * self.t)  # 50 Hz component
            noise = random.gauss(0, 0.2)
            sample = base + noise

            samples_buffer.append(sample)
            self.t += dt

            # If we have enough samples to form a window, push it
            if len(samples_buffer) >= WINDOW_SIZE:
                # Decide randomly if this window is faulty
                is_fault = random.random() < FAULT_PROBABILITY

                samples = np.array(samples_buffer[-WINDOW_SIZE:], dtype=float)

                if is_fault:
                    # Inject some impulses to emulate a bearing defect
                    for _ in range(5):
                        idx = random.randint(0, WINDOW_SIZE - 1)
                        samples[idx] += random.uniform(5, 10)

                window = SensorWindow(
                    timestamp=time.time(),
                    samples=samples,
                    is_fault_injected=is_fault
                )
                self.out_queue.put(window)

                # Remove used samples (sliding window)
                samples_buffer = samples_buffer[int(WINDOW_SIZE / 2):]

            time.sleep(dt)


# ============= DIAGNOSTIC ENGINE ==============

class DiagnosticEngine(threading.Thread):
    """
    Reads SensorWindow objects, computes features, and classifies condition.
    Feature extraction method can be switched between:
      - time      : RMS & Peak
      - cepstrum  : Cepstrum peak index & amplitude
      - hht       : Mean instantaneous amplitude & frequency (Hilbert-based)
    """
    def __init__(self,
                 in_queue: queue.Queue,
                 out_queue: queue.Queue,
                 stop_event: threading.Event,
                 feature_mode: str = "time"):
        super().__init__(daemon=True)
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.feature_mode = feature_mode  # "time", "cepstrum", "hht"

    # --- public API to change feature mode from GUI ---
    def set_feature_mode(self, mode: str):
        if mode in ("time", "cepstrum", "hht"):
            self.feature_mode = mode

    # --- feature extractors ---

    def _features_time(self, samples: np.ndarray):
        rms = float(np.sqrt(np.mean(samples ** 2)))
        peak = float(np.max(np.abs(samples)))
        return rms, peak, "RMS", "Peak"

    def _features_cepstrum(self, samples: np.ndarray):
        """
        Real cepstrum: IFFT(log |FFT|).
        Features:
          - index of maximum cepstral peak (excluding 0)
          - amplitude of that peak.
        """
        spectrum = np.fft.fft(samples)
        log_mag = np.log(np.abs(spectrum) + 1e-12)
        cepstrum = np.fft.ifft(log_mag).real

        cep_abs = np.abs(cepstrum)
        if len(cep_abs) > 1:
            cep_abs[0] = 0.0  # ignore DC term

        peak_idx = int(np.argmax(cep_abs))
        peak_val = float(cep_abs[peak_idx])

        # Feature1: quefrency index (sample index), Feature2: amplitude
        return float(peak_idx), peak_val, "Cep peak idx", "Cep peak amp"

    def _features_hht(self, samples: np.ndarray):
        """
        Simplified HHT-like features using Hilbert transform of the signal:
          - Mean instantaneous amplitude
          - Mean instantaneous frequency
        (No EMD here, just Hilbert of the whole signal.)
        """
        analytic_signal = hilbert_numpy(samples)
        inst_amp = np.abs(analytic_signal)
        phase = np.unwrap(np.angle(analytic_signal))

        if len(phase) < 2:
            return float("nan"), float("nan"), "Mean inst. amp", "Mean inst. freq"

        # instantaneous frequency (Hz)
        inst_freq = np.diff(phase) * SAMPLE_RATE_HZ / (2 * np.pi)

        mean_amp = float(np.mean(inst_amp))
        mean_freq = float(np.mean(inst_freq))

        return mean_amp, mean_freq, "Mean inst. amp", "Mean inst. freq"

    def compute_features(self, samples: np.ndarray):
        mode = self.feature_mode
        if mode == "cepstrum":
            return self._features_cepstrum(samples)
        elif mode == "hht":
            return self._features_hht(samples)
        else:
            return self._features_time(samples)

    def classify(self, samples: np.ndarray) -> (str, str):
        """
        Very simple rule-based classifier based on time-domain RMS/Peak.
        This is intentionally independent of feature_mode so you can
        compare feature sets without changing the classifier.
        """
        rms = float(np.sqrt(np.mean(samples ** 2)))
        peak = float(np.max(np.abs(samples)))

        if peak > 8.0 or rms > 3.0:
            return "FAULT", "High vibration level detected"
        elif peak > 5.0 or rms > 2.0:
            return "WARNING", "Vibration above normal"
        else:
            return "OK", "Normal operation"

    def run(self):
        last_run = 0.0

        while not self.stop_event.is_set():
            now = time.time()
            # Limit diagnostic rate
            if now - last_run < WINDOW_INTERVAL_S:
                time.sleep(0.01)
                continue
            last_run = now

            try:
                window: SensorWindow = self.in_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Extract features according to chosen mode
            feat1, feat2, f1_name, f2_name = self.compute_features(window.samples)
            status, reason = self.classify(window.samples)

            result = DiagnosticResult(
                timestamp=window.timestamp,
                feat1=feat1,
                feat2=feat2,
                feat1_name=f1_name,
                feat2_name=f2_name,
                status=status,
                reason=reason,
                feature_mode=self.feature_mode
            )
            self.out_queue.put(result)


# ============= TKINTER GUI ==============

class PdMSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PdM Simulator on Raspberry Pi")

        # Queues & thread control
        self.sensor_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()

        # Default feature mode
        self.feature_mode_var = tk.StringVar(value="time")

        self.sensor_thread = SensorSimulator(self.sensor_queue, self.stop_event)
        self.diag_thread = DiagnosticEngine(self.sensor_queue,
                                            self.result_queue,
                                            self.stop_event,
                                            feature_mode=self.feature_mode_var.get())

        # GUI layout
        self._build_layout()

        # Start background threads
        self.sensor_thread.start()
        self.diag_thread.start()

        # Periodic UI update
        self.root.after(200, self.poll_results)

    def _build_layout(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Status
        status_frame = ttk.LabelFrame(main_frame, text="Machine Status", padding=10)
        status_frame.grid(row=0, column=0, sticky="ew")

        self.status_label = ttk.Label(status_frame, text="Initializing...",
                                      font=("TkDefaultFont", 16, "bold"))
        self.status_label.grid(row=0, column=0, sticky="w")

        self.reason_label = ttk.Label(status_frame, text="", wraplength=400)
        self.reason_label.grid(row=1, column=0, sticky="w", pady=(5, 0))

        # Features + mode selection
        features_frame = ttk.LabelFrame(main_frame, text="Features", padding=10)
        features_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))

        # Feature mode selector
        ttk.Label(features_frame, text="Feature method:").grid(row=0, column=0, sticky="w")
        mode_combo = ttk.Combobox(
            features_frame,
            textvariable=self.feature_mode_var,
            values=("time", "cepstrum", "hht"),
            state="readonly",
            width=10
        )
        mode_combo.grid(row=0, column=1, sticky="w", padx=(5, 10))
        mode_combo.bind("<<ComboboxSelected>>", self.on_feature_mode_change)

        # Feature labels and values (names will be updated by results)
        self.feature1_label = ttk.Label(features_frame, text="Feature 1:")
        self.feature1_label.grid(row=1, column=0, sticky="w")
        self.feature1_value = ttk.Label(features_frame, text="--")
        self.feature1_value.grid(row=1, column=1, sticky="w")

        self.feature2_label = ttk.Label(features_frame, text="Feature 2:")
        self.feature2_label.grid(row=2, column=0, sticky="w")
        self.feature2_value = ttk.Label(features_frame, text="--")
        self.feature2_value.grid(row=2, column=1, sticky="w")

        # Log
        log_frame = ttk.LabelFrame(main_frame, text="Events", padding=10)
        log_frame.grid(row=2, column=0, sticky="nsew", pady=(10, 0))

        self.log_text = tk.Text(log_frame, height=10, width=60, state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        # Control buttons
        button_frame = ttk.Frame(main_frame, padding=(0, 10, 0, 0))
        button_frame.grid(row=3, column=0, sticky="ew")

        self.quit_button = ttk.Button(button_frame, text="Quit", command=self.on_quit)
        self.quit_button.grid(row=0, column=0, sticky="e")

        main_frame.rowconfigure(2, weight=1)

    # --- callbacks & UI update ---

    def on_feature_mode_change(self, event=None):
        mode = self.feature_mode_var.get()
        self.diag_thread.set_feature_mode(mode)

    def poll_results(self):
        """
        Called periodically to fetch new diagnostic results and update UI.
        """
        try:
            while True:
                result: DiagnosticResult = self.result_queue.get_nowait()
                self.update_status(result)
        except queue.Empty:
            pass

        # Schedule next poll
        self.root.after(200, self.poll_results)

    def update_status(self, result: DiagnosticResult):
        # Update feature labels and values
        self.feature1_label.config(text=f"{result.feat1_name}:")
        self.feature2_label.config(text=f"{result.feat2_name}:")

        self.feature1_value.config(text=f"{result.feat1:.3f}")
        self.feature2_value.config(text=f"{result.feat2:.3f}")

        # Color code status
        if result.status == "OK":
            fg = "green"
        elif result.status == "WARNING":
            fg = "orange"
        else:
            fg = "red"

        self.status_label.config(text=result.status, foreground=fg)
        self.reason_label.config(text=result.reason)

        # Append to log
        ts_str = time.strftime("%H:%M:%S", time.localtime(result.timestamp))
        log_line = (
            f"[{ts_str}] {result.status} "
            f"({result.feature_mode}): "
            f"{result.feat1_name}={result.feat1:.3f}, "
            f"{result.feat2_name}={result.feat2:.3f} - "
            f"{result.reason}\n"
        )
        self.log_text.configure(state="normal")
        self.log_text.insert("end", log_line)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def on_quit(self):
        self.stop_event.set()
        self.root.after(200, self.root.destroy)


if __name__ == "__main__":
    root = tk.Tk()
    app = PdMSimulatorGUI(root)
    root.mainloop()
