import numpy as np
from numpy.random import normal
from main import chirp_bandwidth, chirp_duration, sample_rate, centerFrequency, max_chirps


class RadarChirpSimulator:
    def __init__(self, 
                 B=chirp_bandwidth, T=chirp_duration, f0=centerFrequency, fs=sample_rate,
                 R_target=np.array([10, 100]), 
                 v=np.array([40, -10]), 
                 SNR_dB=-16, 
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
