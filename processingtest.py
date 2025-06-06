import numpy as np
import pyqtgraph as pg
from scipy.fft import fft, fftshift
from numpy.random import normal
from pyqtgraph.Qt import QtCore, QtWidgets
import sys

sample_rate = 1e6
chirp_duration = 0.000128
c = 3e8
num_range_bins = int(chirp_duration * sample_rate)
max_chirps = 255
chirps_per_refresh = 3
chirp_bandwidth = 30e6
centerFrequency = 2.5e9
update_interval = int(chirps_per_refresh * chirp_duration * 1e3)
sweep_count = 0

def next_pow2(n):
    return 1 if n == 0 else 2**int(np.ceil(np.log2(n)))

class RadarChirpSimulator:
    def __init__(self, 
                 B=chirp_bandwidth, T=chirp_duration, f0=centerFrequency, fs=sample_rate,
                 R_target=np.array([20, 100]), 
                 v=np.array([40, -10]), 
                 SNR_dB=-3, 
                 N_slow=max_chirps):
        self.B = B
        self.T = T
        self.f0 = f0
        self.fs = fs
        self.c = 3e8
        self.R_target = R_target
        self.v = v
        self.SNR_dB = SNR_dB
        self.N_slow = N_slow

        self.samples_per_chirp = int(self.T * self.fs)
        self.t = np.arange(0, self.T, 1/self.fs)
        self.k = self.B / self.T
        self.n_sweep = 0

        # TX chirp generation (UNCHANGED)
        chirp = np.exp(-1j * 2 * np.pi * (0.5 * self.k * self.t**2))
        lo = np.exp(-1j * 2 * np.pi * self.f0 * self.t)
        self.s_tx = chirp * lo

    def get_next_chirp(self):
        if self.n_sweep > self.N_slow:
            raise StopIteration("Reached max chirps")

        R_dynamic = self.R_target + self.n_sweep * self.T * self.v
        tau = 2 * R_dynamic / self.c

        s_rx = np.zeros_like(self.t, dtype=np.complex64)
        for delay in tau:
            t_delayed = self.t - delay
            s_rx += np.exp(-1j * 2 * np.pi * (self.f0 * t_delayed + 0.5 * self.k * t_delayed**2))

        signal_power = np.mean(np.abs(s_rx)**2)
        noise_power = signal_power / (10**(self.SNR_dB / 10))
        noise = (np.sqrt(noise_power/2) *
                 (normal(size=s_rx.shape) + 1j * normal(size=s_rx.shape)))
        s_rx += noise

        s_beat = s_rx * np.conj(self.s_tx)

        self.n_sweep += 1
        return s_beat.astype(np.complex64)


def process_and_display():
    global sweep_count
    rows = int(chirp_duration * sample_rate)  # samples per chirp
    cols = max_chirps                        # number of chirps (slow time)
    data_matrix = np.zeros((cols, rows), dtype=np.complex64)  # shape (time, range)

    app = QtWidgets.QApplication(sys.argv)
    win = pg.GraphicsLayoutWidget(show=True, title="Radar Range-Time Response")
    win.resize(800, 600)
    plot = win.addPlot()
    plot.setLabel('left', 'Time Sweep')
    plot.setLabel('bottom', 'Range Bin')
    img_item = pg.ImageItem()
    plot.addItem(img_item)

    lut = pg.colormap.get('viridis').getLookupTable(0.0, 1.0, 256)
    img_item.setLookupTable(lut)
    img_item.setLevels([0, 1])  # required for float images

    sim = RadarChirpSimulator()

    def update():
        global sweep_count, data_matrix
        try:
            new_sweep = sim.get_next_chirp()
        except StopIteration:
            timer.stop()
            print("Max chirps reached.")
            return

        if sweep_count < max_chirps:
            data_matrix[sweep_count, :] = new_sweep
        else:
            data_matrix = np.roll(data_matrix, -1, axis=0)
            data_matrix[-1, :] = new_sweep

        N_FFT = next_pow2(rows)
        FFT_data = fftshift(fft(data_matrix, n=N_FFT, axis=1), axes=1)

        sweep_count += 1

        if sweep_count >= chirps_per_refresh:
            norm_fft = np.abs(FFT_data)
            norm_fft /= np.max(norm_fft)

            # x=range bins, y=time sweeps
            img_item.setImage(norm_fft.T, autoLevels=False, levels=(0, 1))

            # frequency axis for range (beat freq)
            f_axis_range = np.fft.fftfreq(N_FFT, d=1/sample_rate)
            f_axis_range = fftshift(f_axis_range)
            abs_f = np.abs(f_axis_range)
            range_axis = (c * abs_f * chirp_duration) / (2 * chirp_bandwidth)

            elapsed_time = sweep_count * chirp_duration
            window_width = max_chirps * chirp_duration
            start_time = max(0, elapsed_time - window_width)

            img_item.setRect(QtCore.QRectF(0, start_time, range_axis[-1], window_width))
            plot.setLabel('bottom', 'Range [m]')
            plot.setLabel('left', 'Time [s]')

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(update_interval)

    QtWidgets.QApplication.instance().exec_()


if __name__ == '__main__':
    process_and_display()
