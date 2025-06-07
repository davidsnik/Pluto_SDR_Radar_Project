import numpy as np
import pyqtgraph as pg
from scipy.signal import windows
from scipy.fft import fft, fftshift, fftfreq
from numpy.random import normal
from pyqtgraph.Qt import QtCore, QtWidgets
import sys

sample_rate = 1e6     # example value
chirp_duration = 0.000128  # 128 samples if sample_rate is 1 MHz
c = 3e8
num_range_bins = int(chirp_duration * sample_rate)
max_chirps = 255
chirps_per_refresh = 1
chirp_bandwidth = 30e6
centerFrequency = 2.5e9
update_interval = int(chirps_per_refresh*chirp_duration*1e3)
sweep_count = 0
def next_pow2(n):
    return 1 if n == 0 else 2**int(np.ceil(np.log2(n)))


class RadarChirpSimulator:
    def __init__(self, 
                 B=chirp_bandwidth, T=chirp_duration, f0=centerFrequency, fs=sample_rate,
                 R_target=np.array([10, 100]), 
                 v=np.array([40, -10]), 
                 SNR_dB=20, 
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

        # TX chirp
        chirp = np.exp(-1j * 2 * np.pi * (0.5 * self.k * self.t**2))
        lo = np.exp(-1j * 2 * np.pi * self.f0 * self.t)
        self.s_tx = chirp * lo

    def get_next_chirp(self):
        # if self.n_sweep > self.N_slow:
        #     raise StopIteration("Reached max chirps")

        # Compute target positions
        R_dynamic = self.R_target + self.n_sweep * self.T * self.v
        tau = 2 * R_dynamic / self.c  # delay for each target

        # Simulate RX signal
        s_rx = np.zeros_like(self.t, dtype=np.complex64)
        for delay in tau:
            t_delayed = self.t - delay
            s_rx += np.exp(-1j * 2 * np.pi * (self.f0 * t_delayed + 0.5 * self.k * t_delayed**2))

        # Add noise
        signal_power = np.mean(np.abs(s_rx)**2)
        noise_power = signal_power / (10**(self.SNR_dB / 10))
        noise = (np.sqrt(noise_power/2) *
                 (normal(size=s_rx.shape) + 1j * normal(size=s_rx.shape)))
        s_rx += noise

        # Dechirp
        s_beat = s_rx * np.conj(self.s_tx)

        self.n_sweep += 1
        return s_beat.astype(np.complex64)

def ProcessedMatrix():
    global data_matrix, sweep_count
    rows = int(chirp_duration*sample_rate)  # samples per chirp (fast time)
    cols = max_chirps                     # number of chirps (slow time)
    data_matrix = np.zeros((rows, cols), dtype= np.complex64)
    
    N_FFT = next_pow2(num_range_bins)
    f_axis = fftfreq(N_FFT, d=1/sample_rate)
    f_axis = fftshift(f_axis)
    N_FFT2 = next_pow2(max_chirps+1)
    half_idx2 = N_FFT2//2
    f_axis_dop = fftfreq(N_FFT2, d = 1/chirp_duration)
    f_axis_dop_pos = f_axis_dop[half_idx2:]
    # Keep only positive frequencies
    half_idx = N_FFT // 2
    f_axis_pos = f_axis[half_idx:]  # Only positive freqs

    k = chirp_bandwidth / chirp_duration
    range_axis = (c * f_axis_pos) / (2 * k)
    vel_axis = (c * f_axis_dop) / (2 * centerFrequency)
    
    app = QtWidgets.QApplication(sys.argv)
    win1 = pg.GraphicsLayoutWidget(title="Range-Time Plot")
    plot1 = win1.addPlot()
    plot1.setLabel('left', 'Time (s)')
    plot1.setLabel('bottom', 'Range (m)')
    win1.show()

    # Second window
    win2 = pg.GraphicsLayoutWidget(title="Doppler-Range Plot")
    plot2 = win2.addPlot()
    plot2.setLabel('left', 'Velocity (m/s)')
    plot2.setLabel('bottom', 'Range (m)')
    win2.show()
    img_item = pg.ImageItem()
    plot1.addItem(img_item)
    img_doppler = pg.ImageItem()
    # plot2.addItem(img_doppler)

    lut = pg.colormap.get('viridis').getLookupTable(0.0, 1.0, 256)
    img_item.setLookupTable(lut)
    img_item.setLevels([0, 1])  # Fixed amplitude scaling
    img_doppler.setLookupTable(lut)
    img_doppler.setLevels([0,1])
    sim = RadarChirpSimulator()
    def update():
        global sweep_count, data_matrix
        new_sweep = sim.get_next_chirp()

        if sweep_count < max_chirps:
            data_matrix[:, sweep_count] = new_sweep
        else:
            data_matrix[:, :-1] = data_matrix[:, 1:]
            data_matrix[:, -1] = new_sweep

        sweep_count += 1
        
        # Time axis (slow time)
        elapsed_time = sweep_count * chirp_duration
        window_width = max_chirps * chirp_duration
        start_time = max(0, elapsed_time - window_width)
        plot1.setYRange(start_time, start_time + window_width, padding=0)
        
        # FFT along fast time (range)
        FFT_data = fftshift(fft(data_matrix, n=N_FFT, axis=0), axes=0)

        # Only keep positive range bins (shape = (N_FFT//2, N_sweeps))
        FFT_pos = FFT_data[half_idx:, :]  # half_idx = N_FFT // 2

        # Normalize
        norm_fft = np.abs(FFT_pos)
        norm_fft /= np.max(norm_fft)

        # Display range-time spectrogram
        img_item.setImage(norm_fft, autoLevels=False)
        img_item.setRect(QtCore.QRectF(range_axis[0], start_time,
                                    range_axis[-1] - range_axis[0], window_width))

        # Doppler FFT (on slow-time axis)
        # Doppler FFT (slow time -> velocity)
        FFT_2_data = fftshift(fft(FFT_pos, n=N_FFT2, axis=1), axes=1)
        norm_fft_2 = np.abs(FFT_2_data)
        max_val = np.max(norm_fft_2)
        if max_val > 0:
            norm_fft_2 /= max_val
        else:
            norm_fft_2[:] = 0
        img_doppler.setImage(norm_fft_2, autoLevels=False)

        img_doppler.setRect(QtCore.QRectF(
            range_axis[0], vel_axis[0],
            range_axis[-1] - range_axis[0], vel_axis[-1] - vel_axis[0]
        ))


    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(update_interval)

    QtWidgets.QApplication.instance().exec_()


if __name__ == '__main__':
    ProcessedMatrix()
