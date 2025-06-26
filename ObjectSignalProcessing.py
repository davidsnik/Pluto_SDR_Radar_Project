import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import scipy.fft as fft
# from main import chirp_bandwidth, chirp_duration, centerFrequency, sample_rate, c
import matplotlib.pyplot as plt
from processingtest import RadarChirpSimulator
from matplotlib.animation import FuncAnimation
import scipy.io
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
import pyqtgraph.exporters
# chirp_bandwidth = 30e6 # hz
# chirp_duration = 1 # ms
# max_chirps = 10
# centerFrequency = 2.5e9 # in hz
# c = 3e8
# sample_rate = 60500000 # in hz
chirp_bandwidth = 200e6 # hz
chirp_duration = 1e-4 # ms
max_chirps = 255
centerFrequency = 2e9 # in hz
c = 3e8
sample_rate = 5e8 # in hz

training_cells = 12
guard_cells = 10
PFA = 1e-4

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
        self.rows = int(self.chirp_duration*self.sample_rate)
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
        f_axis = fft.fftshift(np.fft.fftfreq(self.N_FFT, d=1/self.sample_rate))
        f_axis = f_axis[self.pos_indices:]  # take only positive bins after shift
        range_axis = (c * f_axis) / (2 * self.k)
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
    
    def CA_CFAR(self, data_matrix: np.ndarray, guard_cells: int, training_cells: int, PFA: float):
        """CA-CFAR detection algorithm

        Args:
            data_matrix (np.ndarray): Range-time data matrix of shape (samples_per_chirp, max_chirps)
            guard_cells (int): Amount of guard cells around the CUT
            training_cells (int): Amount of cells taken into account for the averaging
            PFA (float): Probability of false alarm

        Returns:
            detection_bin (list): Range bins of detected targets (range_axis[detection_bin] gives the detection distance)
        """
        padding = training_cells + guard_cells
        total_cells = 1 + 2*padding
        if self.count < self.max_chirps:
            edge_padded_signal = np.pad(data_matrix[self.count-1,:], (padding, padding), mode = 'edge')
            window = sliding_window_view(edge_padded_signal, total_cells) #args: array to be taken into consideration, window shape
        
            mask = np.ones(total_cells, dtype= bool)
            mask[training_cells:training_cells+2*guard_cells+1] = False
            
            avg_noise = np.sum(window[:,mask], axis = 1)/np.sum(mask)
            
            alpha = training_cells*(PFA**(-1/training_cells)-1)
            threshold = alpha * avg_noise
            
            detection_bin = data_matrix[self.count-1,:] > threshold
            return detection_bin
        else:
            edge_padded_signal = np.pad(data_matrix[self.max_chirps-1,:], (padding, padding), mode = 'edge')
            window = sliding_window_view(edge_padded_signal, total_cells) #args: array to be taken into consideration, window shape
            
            mask = np.ones(total_cells, dtype= bool)
            mask[training_cells:training_cells+2*guard_cells+1] = False
            
            avg_noise = np.sum(window[:,mask], axis = 1)/np.sum(mask)
            
            alpha = training_cells*(PFA**(-1/training_cells)-1)
            threshold = alpha * avg_noise
            
            detection_bin = data_matrix[self.max_chirps-1,:] > threshold
            return detection_bin
        
import sys
import numpy as np
import scipy.fft as fft
from numpy.lib.stride_tricks import sliding_window_view
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore

# (Your existing DoubleFFT and other functions remain unchanged)
# I assume DoubleFFT, RadarChirpSimulator, etc. are already defined

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, double_fft_instance, sim_instance, s_beat_matrix, range_axis, vel_axis, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Radar Visualization Main Window")
        self.resize(1600, 600)

        self.test = double_fft_instance
        self.sim = sim_instance
        self.s_beat_matrix = s_beat_matrix

        # Central widget and layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        layout = QtWidgets.QHBoxLayout()
        central_widget.setLayout(layout)

        # Setup plots as GraphicsLayoutWidgets (containers)
        self.win_range = pg.GraphicsLayoutWidget()
        self.win_range.setWindowTitle("Real-Time Range-Time Plot")
        self.win_range.setBackground('w')
        layout.addWidget(self.win_range)

        self.win_doppler = pg.GraphicsLayoutWidget()
        self.win_doppler.setWindowTitle("Range-Doppler Plot")
        self.win_doppler.setBackground('w')
        layout.addWidget(self.win_doppler)

        self.win_freq = pg.GraphicsLayoutWidget()
        self.win_freq.setWindowTitle("Frequency Response & CFAR Threshold")
        self.win_freq.setBackground('w')
        layout.addWidget(self.win_freq)

        # Setup Range-Time plot inside win_range
        self.plot_range = self.win_range.addPlot(title="Range-Time")


        black_pen = pg.mkPen('k')
        self.plot_range.getAxis('bottom').setPen(black_pen)
        self.plot_range.getAxis('bottom').setTextPen(black_pen)
        self.plot_range.getAxis('left').setPen(black_pen)
        self.plot_range.getAxis('left').setTextPen(black_pen)

        self.plot_range.setTitle("Range-Time", color='k')
        self.img1 = pg.ImageItem()
        
        cmap = pg.colormap.get('viridis')
        lut = cmap.getLookupTable(0.0, 1.0, 256)
        self.img1.setLookupTable(lut)
        self.plot_range.addItem(self.img1)
        self.plot_range.setLabel('left', 'Time', units='s', color = 'k')
        self.plot_range.setLabel('bottom', 'Range', units='m', color = 'k')
        self.plot_range.setRange(xRange=[0, 800])

        # Setup Doppler plot inside win_doppler
        self.plot_doppler = self.win_doppler.addPlot(title='Range-Doppler', color = 'k')
        self.img2 = pg.ImageItem()
        self.img2.setLookupTable(lut)
        self.plot_doppler.addItem(self.img2)
        self.plot_doppler.setLabel('left', 'Velocity', units='m/s')
        self.plot_doppler.setLabel('bottom', 'Range', units='m')
        self.plot_doppler.setRange(xRange=[0, 800])


        self.plot_doppler.getAxis('bottom').setPen(black_pen)
        self.plot_doppler.getAxis('bottom').setTextPen(black_pen)
        self.plot_doppler.getAxis('left').setPen(black_pen)
        self.plot_doppler.getAxis('left').setTextPen(black_pen)
        self.plot_doppler.setTitle("Range-Doppler", color='k')
        
        # Setup Frequency + CFAR plot inside win_freq
        self.plot_freq = self.win_freq.addPlot(title="Chirp FFT and CFAR Threshold", color = 'k')
        self.curve_fft = self.plot_freq.plot(pen='b', name="FFT Magnitude", color = 'k')
        self.curve_thresh = self.plot_freq.plot(pen='r', name="CFAR Threshold", color = 'k')
        self.plot_freq.setLabel('left', 'Amplitude', color = 'k')
        self.plot_freq.setLabel('bottom', 'Range', units='m', color = 'k')
        self.plot_freq.setLogMode(False, True)
        self.plot_freq.setRange(xRange=[0, 800])


        self.plot_freq.getAxis('bottom').setPen(black_pen)
        self.plot_freq.getAxis('bottom').setTextPen(black_pen)
        self.plot_freq.getAxis('left').setPen(black_pen)
        self.plot_freq.getAxis('left').setTextPen(black_pen)
        self.plot_freq.setTitle("Chirp FFT and CFAR Threshold", color='k')
        # Initialize variables
        self.i = 0
        self.paused = False

        # Timer for updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(30)  # milliseconds

    def toggle_pause(self):
        self.paused = not self.paused
        print("Paused" if self.paused else "Resumed")

    def save_screenshots(self):
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exporter1 = pg.exporters.ImageExporter(self.plot_range)
        exporter1.export(f"range_time_{timestamp}.png")

        exporter2 = pg.exporters.ImageExporter(self.plot_doppler)
        exporter2.export(f"range_doppler_{timestamp}.png")

        exporter3 = pg.exporters.ImageExporter(self.plot_freq)
        exporter3.export(f"fft_threshold_{timestamp}.png")

        print(f"Screenshots saved with timestamp {timestamp}")
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_S:
            print("S key pressed — saving screenshots.")
            self.save_screenshots()
    def update(self):
        if self.paused:
            return

        if self.i >= self.s_beat_matrix.shape[0]:
                print("Reached end of data — stopping updates.")
                self.timer.stop()
                return

        chirp = self.s_beat_matrix[self.i, :]
        # chirp = self.sim.get_next_chirp()

        range_matrix, time_axis, range_axis = self.test.get_range_time(chirp[:-1])

        # Update Range-Time plot
        self.img1.setImage(range_matrix.T, autoLevels=(0, 1))
        self.img1.setRect(pg.QtCore.QRectF(
            range_axis[0], time_axis[0],
            range_axis[-1] - range_axis[0],
            time_axis[-1] - time_axis[0]
        ))

        detect_bins = self.test.CA_CFAR(data_matrix=range_matrix,
                                        guard_cells=10,
                                        training_cells=12,
                                        PFA=1e-4)
        print(f'Targets at {range_axis[detect_bins]}')

        # Update Doppler only if enough chirps collected
        if self.test.count >= self.test.velocity_buffer_size:
            doppler_matrix, vel_axis, range_axis_dop = self.test.get_range_doppler()
            self.img2.setImage(doppler_matrix.T, autoLevels=(0, 1))
            self.img2.setRect(pg.QtCore.QRectF(
                range_axis_dop[0], vel_axis[0],
                range_axis_dop[-1] - range_axis_dop[0],
                vel_axis[-1] - vel_axis[0]
            ))

        self.i += 1

        # Frequency response + CFAR Threshold for current chirp
        padding = 10 + 12
        total_cells = 1 + 2 * padding

        current_fft = np.abs(fft.fftshift(fft.fft(chirp, n=self.test.N_FFT)))[self.test.pos_indices:]
        current_fft /= np.max(current_fft)
        range_axis_fft = self.test.get_range_axis()

        edge_padded = np.pad(current_fft, (padding, padding), mode='edge')
        window = sliding_window_view(edge_padded, total_cells)

        mask = np.ones(total_cells, dtype=bool)
        mask[12:12 + 2*10 + 1] = False

        avg_noise = np.sum(window[:, mask], axis=1) / np.sum(mask)
        alpha = 12 * (1e-4 ** (-1 / 12) - 1)
        threshold = alpha * avg_noise

        self.curve_fft.setData(range_axis_fft, current_fft)
        self.curve_thresh.setData(range_axis_fft, threshold)
            


if __name__ == '__main__':
    import scipy.io
    raw_data = scipy.io.loadmat(r"C:\Users\L3PyT\OneDrive\Documents\GitHub\Pluto_SDR_Radar_Project\Test Data\Scene4.mat")
    samples_per_chirp = len(raw_data['tx_waveform'])

    s_tx = np.array(raw_data['tx_waveform']).T.reshape(-1,)
    s_tx = np.tile(s_tx, int(len(raw_data['received_data'].squeeze().flatten())/samples_per_chirp)).astype(np.complex64)
    s_tx *= raw_data['tx_gain'].squeeze()
    s_raw = raw_data['received_data'].squeeze().astype(np.complex64).T.reshape(-1, 1)
    s_beat_vector = (np.squeeze(s_raw) * np.conj(s_tx))
    s_beat_matrix = np.reshape(s_raw, (samples_per_chirp, -1)).T

    sim = RadarChirpSimulator()
    chirp_bandwidth = 200e6
    chirp_duration = 1e-4
    centerFrequency = 2e9
    sample_rate = 5e8
    sample_rate = raw_data['fs'].squeeze()
    # test = DoubleFFT(chirp_bandwidth=chirp_bandwidth,
    #                  chirp_duration=chirp_duration,
    #                  center_frequency=centerFrequency,
    #                  sample_rate=sample_rate, max_chirps=256,
    #                  velocity_buffer_size=128)
    test = DoubleFFT(chirp_bandwidth=raw_data['fstop'].squeeze()-raw_data['fstart'].squeeze(),
                    chirp_duration= raw_data['chirp_duration'].squeeze(),
                    center_frequency=raw_data['centerFrequency'].squeeze(),
                    sample_rate=sample_rate, max_chirps=10,
                    velocity_buffer_size=10)

    app = QtWidgets.QApplication([])
    main_window = MainWindow(test, sim, s_beat_matrix, None, None)
    main_window.show()
    app.exec_()
