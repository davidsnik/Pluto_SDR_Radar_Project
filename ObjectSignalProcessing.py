import numpy as np
import scipy.fft as fft
from main import chirp_bandwidth, chirp_duration, centerFrequency, sample_rate, c
import matplotlib.pyplot as plt
from processingtest import RadarChirpSimulator


def next_pow2(n):
    return 1 if n == 0 else 2**int(np.ceil(np.log2(n)))

class DoubleFFT:
    def __init__(self, chirp_bandwidth: int, chirp_duration: float, center_frequency: int, sample_rate: int, max_chirps: int, velocity_buffer_size: int):
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
        self.velocity_buffer_size = velocity_buffer_size
        self.N_FFT = next_pow2(self.rows)
        self.N_Doppler = next_pow2(self.velocity_buffer_size)
        #self.N_Doppler= next_pow2(self.cols)
        self.pos_indices = self.N_FFT//2
        self.range_fft_buffer = None


    def get_range_matrix(self, data_vector: np.ndarray):
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
        range_matrix = np.abs(Non_normalized_FFT_data) / np.max(np.abs(Non_normalized_FFT_data))
        
        return range_matrix.T
    
    def get_range_axis(self):
            f_axis = fft.fftshift(fft.fftfreq(self.N_FFT, d=1/sample_rate))
            f_axis_pos = f_axis[self.pos_indices:] 
            range_axis = (c * f_axis_pos) / (2 * self.k)
            return range_axis
    
    def get_time_axis(self):
        if self.count <= self.max_chirps:
            # Haven't filled buffer yet
            time_axis = np.arange(self.max_chirps) * self.chirp_duration
        else:
            # Buffer is full, show time for the most recent max_chirps
            start_chirp = self.count - self.max_chirps
            time_axis = (np.arange(start_chirp, start_chirp + self.max_chirps) * self.chirp_duration)
        return time_axis

    def get_velocity_matrix(self):
        if self.range_fft_buffer is None:
            raise RuntimeError("Call get_matrix() first to compute range FFT.")
        #This ifelse maks sure slow_time_profiles always chooses something between 0 and 255, since FFT_pos has shape (..,255)
        if self.count<=self.cols:
            start = max(0, self.count - self.velocity_buffer_size) #is used so we don't select negative array index
            slow_time_profiles = self.range_fft_buffer[:,start : start+ self.velocity_buffer_size] #Choose last N_doppler chirps to apply FFT on
        elif self.count>self.cols:
                slow_time_profiles = self.range_fft_buffer[:, self.cols-self.velocity_buffer_size : self.cols]
        elif self.count < self.velocity_buffer_size:
            raise RuntimeError(f"Need at least {self.velocity_buffer_size} chirps, have {self.count}")

        doppler_not_normalized = fft.fftshift(fft.fft(slow_time_profiles, n = self.N_Doppler, axis=1), axes=1)
        velocity_range_matrix= np.abs(doppler_not_normalized)/np.max(np.abs(doppler_not_normalized))
        return velocity_range_matrix.T
   
    def get_velocity_axis(self):
        N_FFT_doppler = next_pow2(self.N_Doppler)
        f_axis_dop = np.fft.fftfreq(N_FFT_doppler, d=self.chirp_duration)
        f_axis_dop= fft.fftshift(f_axis_dop)
        vel_axis = (c * f_axis_dop) / (2 * self.center_frequency)
        return vel_axis

    def get_range_time(self, data_vector: np.ndarray):
        """Gives everything needed for a range-time plot

        Args:
            data_vector (np.ndarray): Vector containing one chirp of size (samples_per_chirp, 1)

        Returns:
            range_matrix: Range-time matrix of shape (max_chirps, range_bins)
            time_axis: Time axis of shape (max_chirps, 1)
            range_axis: Range axis of shape (range_bins, 1)
        """
        range_matrix = self.get_range_matrix(data_vector)
        time_axis = self.get_time_axis()
        range_axis = self.get_range_axis()
        
        return range_matrix, time_axis, range_axis
    
    def get_range_doppler(self):
        """
        Returns:
        Organiseert alles wat te maken heeft met velocity estimation. Dan kan in principe alleen deze
        functie opgeroepen worden.
          doppler_matrix: shape (n_doppler_bins, n_range_bins)
          vel_axis:       shape (n_doppler_bins,) [m/s]
          range_axis:     shape (n_range_bins,) [meters]
        """
        # ensure we have enough chirps
        if self.count < self.velocity_buffer_size:
            raise RuntimeError(f"Need {self.velocity_buffer_size} chirps for Doppler, have {self.count}")

        doppler_matrix = self.get_velocity_matrix()
        vel_axis = self.get_velocity_axis()
        range_axis = self.get_range_axis()
        return doppler_matrix, vel_axis, range_axis

def plot_range_doppler(range_matrix, vel_matrix, range_axis, vel_axis, chirp_duration, time_axis):
    """
    range_matrix:   2D np.array, shape (n_chirps,   n_range_bins)
    vel_matrix:     2D np.array, shape (n_range_bins, n_doppler_bins)
    range_axis:     1D np.array, length = n_range_bins (meters)
    vel_axis:       1D np.array, length = n_doppler_bins (m/s)
    chirp_duration: scalar, seconds per chirp
    """
    # Build time axis for range-time plot


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    # ——— Range-Time plot ——————————————————————————————————————
    im1 = ax1.pcolormesh(
        range_axis,      # x
        time_axis,       # y
        range_matrix,    # C (shape = [len(y), len(x)])
        shading='auto'
    )
    ax1.set_title("Range–Time")
    ax1.set_xlabel("Range (m)")
    ax1.set_ylabel("Time (s)")
    ax1.set_xlim(0,200)
    fig.colorbar(im1, ax=ax1, label="Normalized amplitude")

    # ——— Range–Doppler plot ————————————————————————————————————
    im2 = ax2.pcolormesh(
        range_axis,      # x
        vel_axis,        # y
        vel_matrix,    # C (shape = [len(y), len(x)])
        shading='auto'
    )
    ax2.set_title("Range–Doppler")
    ax2.set_xlabel("Range (m)")
    ax2.set_ylabel("Velocity (m/s)")
    ax2.set_xlim(0,200)
    fig.colorbar(im2, ax=ax2, label="Normalized amplitude")

    plt.show()

sim = RadarChirpSimulator()
test = DoubleFFT(chirp_bandwidth=chirp_bandwidth, chirp_duration= chirp_duration, center_frequency=centerFrequency,sample_rate=sample_rate, max_chirps=10, velocity_buffer_size=8)
i = 0       

##THIS ONE WORKS
if __name__ == '__main__':
    while i < 16:
        s = sim.get_next_chirp()
        range_matrix, time_axis, range_axis = test.get_range_time(s)
        # Once enough chirps, compute Doppler and plot
        #Dit is nodig, omdat er minimaal velocity_buffer_size chirps nodig zijn voor velocity estimation.
        if test.count >= test.velocity_buffer_size:
            doppler_matrix,velocity_axis,range_axis=test.get_range_doppler()
            plot_range_doppler(
                range_matrix=range_matrix,
                vel_matrix =doppler_matrix,
                range_axis =range_axis,
                vel_axis   =velocity_axis,
                chirp_duration=chirp_duration,
                time_axis = time_axis
            )

        i += 1



##Dit plakt elke frame achter elkaar. Kan crash verorozaken
# plt.ion()
# fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,6), constrained_layout=True)

# for i in range(200):
#     # 1) Always get the next beat-note and update range
#     beat = sim.get_next_chirp()
#     full_range = test.get_range_matrix(beat)    # shape = (n_filled, n_range_bins)
#     n_filled   = min(test.count, test.max_chirps)
#     range_matrix = full_range[:n_filled, :]

#     # 2) Only do Doppler once the slow-time buffer is full
#     if test.count >= test.velocity_buffer_size:
#         vel_matrix = test.get_velocity_matrix()    # shape = (n_doppler_bins, n_range_bins)
#         r_axis     = test.get_range_axis()         # len = n_range_bins
#         v_axis     = test.get_velocity_axis()      # len = n_doppler_bins

#         # 3) Clear old plots
#         ax1.cla()
#         ax2.cla()

#         # 4) Range–Time
#         time_axis = np.arange(n_filled) * chirp_duration
#         ax1.pcolormesh(r_axis, time_axis, range_matrix, shading='auto')
#         ax1.set_ylabel("Time (s)")
#         ax1.set_xlabel("Range (m)")
#         ax1.set_title("Range–Time")
#         ax1.set_xlim(0,120)
#         # 5) Range–Doppler
#         ax2.pcolormesh(r_axis, v_axis, vel_matrix, shading='auto')
#         ax2.set_ylabel("Velocity (m/s)")
#         ax2.set_xlabel("Range (m)")
#         ax2.set_title("Range–Doppler")
#         ax2.set_xlim(0,120)
#         # 6) Redraw
#         plt.pause(0.01)

# # Turn off interactive mode if you like
# plt.ioff()
