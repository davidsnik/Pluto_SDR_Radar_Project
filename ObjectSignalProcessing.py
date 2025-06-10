import numpy as np
import scipy.fft as fft
from main import chirp_bandwidth, chirp_duration, centerFrequency, sample_rate, c
import matplotlib.pyplot as plt
from processingtest import RadarChirpSimulator


def next_pow2(n):
    return 1 if n == 0 else 2**int(np.ceil(np.log2(n)))

class DoubleFFT:
    def __init__(self, chirp_bandwidth: int, chirp_duration: float, center_frequency: int, sample_rate: int, max_chirps: int):
        self.chirp_bandwidth = chirp_bandwidth
        self.chirp_duration = chirp_duration
        self.center_frequency = center_frequency
        self.sample_rate = sample_rate
        self.max_chirps = max_chirps
        self.k = self.chirp_bandwidth/self.chirp_duration
        self.rows = int(chirp_duration*sample_rate)
        self.cols = max_chirps
        self.data_matrix = np.zeros((self.rows, self.cols), dtype = np.complex64)
        self.count = 0
        self.N_FFT = next_pow2(self.rows)
        self.pos_indices = self.N_FFT//2
        self.range_fft_buffer = None

    def get_matrix(self, data_vector: np.ndarray):
        if data_vector.ndim != 1 or data_vector.size != self.rows:
            raise ValueError(f"Expected 1D array of length {int(self.rows)}, got shape {tuple(data_vector.shape)}")
        data_vector = data_vector.reshape(self.rows, 1)

        if self.count < self.max_chirps:
            self.data_matrix[:, self.count] = data_vector[:, 0]
        else:
            self.data_matrix = np.roll(self.data_matrix, -1, axis=1)
            self.data_matrix[:, -1] = data_vector[:, 0]
        self.count += 1

        pos_indices =  self.N_FFT // 2

        Non_normalized_FFT_data = fft.fftshift(fft.fft(self.data_matrix, n=self.N_FFT, axis=0), axes=0)[self.pos_indices:]
        self.range_fft_buffer = Non_normalized_FFT_data
        Range_matrix = np.abs(Non_normalized_FFT_data) / np.max(np.abs(Non_normalized_FFT_data))
        
        return Range_matrix.T
    def get_range_axis(self):
            f_axis = fft.fftshift(fft.fftfreq(self.N_FFT, d=1/sample_rate))
            f_axis_pos = f_axis[self.pos_indices:] 
            range_axis = (c * f_axis_pos) / (2 * self.k)
            return range_axis
    def get_velocity_matrix(self,N_Doppler):
        if self.range_fft_buffer is None:
            raise RuntimeError("Call get_matrix() first to compute range FFT.")

        if self.count >= N_Doppler:
            #This ifelse maks sure slow_time_profiles always chooses something between 0 and 255, since FFT_pos has shape (..,255)
            if self.count<=self.cols:
                slow_time_profiles = self.range_fft_buffer[:, self.count-N_Doppler : self.count] #Choose last N_doppler chirps to apply FFT on
            else:
                slow_time_profiles = self.range_fft_buffer[:, self.cols-N_Doppler : self.cols]
        N_FFT_doppler = next_pow2(N_Doppler)
        doppler_not_normalized = fft.fftshift(fft(slow_time_profiles, axis=1), axes=1)
        doppler_normalized= np.abs(doppler_not_normalized)/np.max(np.abs(doppler_not_normalized))
        return doppler_normalized
    def get_velocity_axis(self,N_Doppler):
        N_FFT_doppler = next_pow2(N_Doppler)
        f_axis_dop = np.fft.fftfreq(N_FFT_doppler, d=self.chirp_duration)
        f_axis_dop=np.fft.fftshift(f_axis_dop)
        vel_axis = (c * f_axis_dop) / (2 * self.center_frequency)
        return vel_axis

 
sim = RadarChirpSimulator()
test = DoubleFFT(chirp_bandwidth=chirp_bandwidth, chirp_duration= chirp_duration, center_frequency=centerFrequency,sample_rate=sample_rate, max_chirps=256)
i = 0       

if __name__ == '__main__':
    while i < 10:
        next_chirp = sim.get_next_chirp()
        real = test.get_matrix(next_chirp)
        axis = test.get_range_axis()
        samples = range(256)
        plt.pcolormesh(axis,samples, real)
        plt.xlim((0,200))
        plt.show()
        i +=1
