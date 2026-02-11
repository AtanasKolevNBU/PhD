import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

class Cepstrum:

    def __init__(self):
        pass

    def cepstrum_calculation(self, signal):
        spectrum = np.fft.fft(signal)
        log_spectrum = np.log(np.abs(spectrum) + np.finfo(float).eps)
        cepstrum = np.fft.ifft(log_spectrum).real

        freqs = np.fft.fftfreq(len(log_spectrum), 1/12000)
        quefrencies = np.fft.fftfreq(len(cepstrum), 1/12000)

        return cepstrum, log_spectrum, freqs, quefrencies
    
    def cepstrum_visualizer_plotly(self, signal, name):
        signal_cepstrum, signal_log_spectrum, signal_freq, signal_quefreq = self.cepstrum_calculation(signal)
        lifter = np.ones_like(signal_cepstrum)
        lifter[:30] = 0
        fig = make_subplots(rows=3, cols=1, subplot_titles=(f'Сигнал - {name}', 'Log спектър', 'Cepstrum'))

        # Add the original time series plot
        fig.add_trace(go.Scatter(x=signal.index, y=signal, mode='lines', name='Сигнал'), row=1, col=1)

        fig.add_trace(go.Scatter(x=signal_freq[:len(signal_freq)//2], y=signal_log_spectrum[:len(signal_log_spectrum)//2], mode='lines', name='Log спектър'), row = 2, col = 1)

        # Add the FFT magnitude plot
        fig.add_trace(go.Scatter(x=signal_quefreq[:len(signal_quefreq)//2], y=signal_cepstrum[:len(signal_cepstrum)//2]*lifter[:len(lifter)//2] , mode='lines', name='Амплитуда на cepstrum'), row=3, col=1)

        # Update layout
        fig.update_xaxes(title_text='Време', row=1, col=1)
        fig.update_yaxes(title_text='Стойност', row=1, col=1)


        fig.update_xaxes(title_text='Честота (Hz)', row=2, col=1)
        fig.update_yaxes(title_text='Амплитуда', row=2, col=1)

        fig.update_xaxes(title_text='Q - Честота (s)', row=3, col=1)
        fig.update_yaxes(title_text='Амплитуда', row=3, col=1)
        fig.update_layout(height=800, width=1500, title_text='Сигнал, Log спектър и Cepstrum')

        return fig
    
    def plot_signal_and_cepstrum(self,
        x: np.ndarray,
        fs: float,
        global_title:str = 'Cepstrum-based analysis of vibration signal',
        title_signal: str = "Raw signal (time domain)",
        title_cepstrum: str = "Real cepstrum (quefrency domain)",
        quef_max_s: float | None = None,
        min_quef_s: float = 1e-3,
        y_lim_bottom: float = -0.2,
        y_lim_top:float = 0.2
    ) -> None:
        """
        Plot raw signal and its cepstrum in two stacked subplots.

        Parameters
        ----------
        x : np.ndarray
            1D time-domain signal.
        fs : float
            Sampling frequency [Hz].
        quef_max_s : float | None
            If set, limit cepstrum x-axis to [0, quef_max_s] seconds.
            Useful to focus on physically meaningful quefrency range.
        """
        x = np.asarray(x, dtype=float).ravel()
        n = x.size
        if n < 2:
            raise ValueError("Signal must have at least 2 samples.")
        if fs <= 0:
            raise ValueError("fs must be > 0.")

        t = np.arange(n) / fs
        c, _, _, quefrencies = self.cepstrum_calculation(x)

        q = quefrencies

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
        fig.suptitle(global_title, fontsize=18)

        # --- Subplot 1: raw signal ---
        axes[0].plot(t, x)
        axes[0].set_title(title_signal, fontsize=16)
        axes[0].set_xlabel("Time [s]", fontsize=14)
        axes[0].set_ylabel("Amplitude", fontsize=14)
        axes[0].tick_params(labelsize=12)
        axes[0].grid(True)

        # --- Subplot 2: cepstrum ---
        if quef_max_s is not None:
            if quef_max_s <= 0:
                raise ValueError("quef_max_s must be > 0.")
            max_idx = int(min(n, np.floor(quef_max_s * fs)))
            max_idx = max(max_idx, 2)  # keep at least a couple points
            q_plot = q[:max_idx]
            c_plot = c[:max_idx]
        else:
            half_n = n // 2
            q_plot = q[:half_n]
            c_plot = c[:half_n]
        print(q_plot.shape)
        print(c_plot.shape)

        axes[1].plot(q_plot, c_plot)
        axes[1].set_title(title_cepstrum, fontsize=16)
        axes[1].set_xlabel("Quefrency [s]", fontsize=14)
        axes[1].set_ylabel("Cepstrum amplitude", fontsize=14)
        axes[1].set_ylim(bottom = y_lim_bottom, top = y_lim_top)
        axes[1].tick_params(labelsize=12)
        axes[1].grid(True)

        fig.tight_layout()
        plt.show()