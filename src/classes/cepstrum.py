import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

class Cepstrum:

    def __init__(self):
        pass

    def cepstrum_calculation(signal):
        spectrum = np.fft.fft(signal)
        log_spectrum = np.log(np.abs(spectrum) + np.finfo(float).eps)
        cepstrum = np.fft.ifft(log_spectrum).real

        freqs = np.fft.fftfreq(len(log_spectrum), 1/12000)
        quefrencies = np.fft.fftfreq(len(cepstrum), 1/12000)

        return cepstrum, log_spectrum, freqs, quefrencies
    
    def cepstrum_visualizer(self, signal, name):
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