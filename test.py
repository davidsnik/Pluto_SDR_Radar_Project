import numpy as np
import scipy.io
import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer
from processingtest import RadarChirpSimulator
from ObjectSignalProcessing import DoubleFFT  # Your code above should be in a module or same file

# chirp_bandwidth = 30e6
# chirp_duration = 1
# max_chirps = 10
# centerFrequency = 2.5e9
# sample_rate = 60500000
velocity_buffer_size = 20
chirp_bandwidth = 30e6 # hz
chirp_duration = 0.000128 # ms
max_chirps = 255
centerFrequency = 2.5e9 # in hz
c = 3e8
sample_rate = 60.5e6 # in hz

class RadarApp(QtWidgets.QMainWindow):
    def __init__(self, s_beat_matrix, range_processor):
        super().__init__()
        self.setWindowTitle("Radar Range-Time and Range-Doppler Viewer")
        self.s_beat_matrix = s_beat_matrix
        self.range_processor = range_processor
        self.index = 0

        # Plot widgets
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.plot_widget)

        # Range-Time plot
        self.range_plot = self.plot_widget.addPlot(title="Range-Time")
        self.range_img = pg.ImageItem()
        self.range_plot.addItem(self.range_img)
        self.range_plot.setLabel('left', 'Time', units='s')
        self.range_plot.setLabel('bottom', 'Range', units='m')

        # Range-Doppler plot
        self.plot_widget.nextRow()
        self.doppler_plot = self.plot_widget.addPlot(title="Range-Doppler")
        self.doppler_img = pg.ImageItem()
        self.doppler_plot.addItem(self.doppler_img)
        self.doppler_plot.setLabel('left', 'Velocity', units='m/s')
        self.doppler_plot.setLabel('bottom', 'Range', units='m')

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1)  # ms

    def update(self):
        if self.index >= len(self.s_beat_matrix):
            return

        chirp = self.s_beat_matrix #[self.index, :]
        range_matrix, time_axis, range_axis = self.range_processor.get_range_time(chirp)

        # Update Range-Time image
        self.range_img.setImage(range_matrix.T, autoLevels=(0,1))
        self.range_img.setRect(pg.QtCore.QRectF(range_axis[0], time_axis[0],range_axis[-1] - range_axis[0], time_axis[-1] - time_axis[0]))

        # If enough data, update Range-Doppler
        if self.range_processor.count >= velocity_buffer_size:
            doppler_matrix, vel_axis, range_axis = self.range_processor.get_range_doppler()
            self.doppler_img.setImage(doppler_matrix.T, autoLevels=(0,1))
            self.doppler_img.setRect(pg.QtCore.QRectF(range_axis[0], vel_axis[0], range_axis[-1] - range_axis[0], vel_axis[-1] - vel_axis[0]))

        self.index += 1

def main():
    raw_data = scipy.io.loadmat(r"C:\Users\L3PyT\OneDrive\Documents\GitHub\Pluto_SDR_Radar_Project\Test Data\Scene3.mat")
    samples_per_chirp = len(raw_data['tx_waveform'])
    
    s_tx = np.array(raw_data['tx_waveform']).T.reshape(-1,)
    s_tx = np.tile(s_tx, int(len(raw_data['received_data'].squeeze().flatten())/samples_per_chirp)).astype(np.complex64)
    s_tx *= raw_data['tx_gain'].squeeze()

    s_raw = raw_data['received_data'].squeeze().astype(np.complex64).T.reshape(-1, 1)
    s_beat_vector = (np.squeeze(s_raw) * np.conj(s_tx))
    s_beat_matrix = np.reshape(s_beat_vector, (samples_per_chirp, -1)).T

    sim = RadarChirpSimulator()
    
    data = sim.get_next_chirp()
    processor = DoubleFFT(
        chirp_bandwidth=chirp_bandwidth,
        chirp_duration=chirp_duration,
        center_frequency=centerFrequency,
        sample_rate=raw_data['fs'][0][0],
        max_chirps=max_chirps,
        velocity_buffer_size=velocity_buffer_size
    )

    app = QtWidgets.QApplication([])
    radar_viewer = RadarApp(data, processor)
    radar_viewer.show()
    app.exec_()

if __name__ == "__main__":
    main()
