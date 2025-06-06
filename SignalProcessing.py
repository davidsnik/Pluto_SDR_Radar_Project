import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import sys

update_interval = 50  # ms
sample_rate = 1e6     # example value
chirp_duration = 0.000128  # 128 samples if sample_rate is 1 MHz

num_range_bins = int(chirp_duration * sample_rate)
max_chirps = 200
chirps_per_refresh = 3

data_matrix = np.zeros((num_range_bins, max_chirps))
sweep_count = 0

def get_radar_sweep():
    sweep = np.exp(-((np.arange(num_range_bins) - 50)**2) / (2 * 10**2))
    sweep += 0.2 * np.random.randn(num_range_bins)
    return sweep

def ProcessedMatrix():
    global data_matrix, sweep_count

    app = QtWidgets.QApplication(sys.argv)
    win = pg.GraphicsLayoutWidget(show=True, title="Radar Range-Time Response")
    win.resize(800, 600)
    plot = win.addPlot()
    plot.setLabel('left', 'Range Bin')
    plot.setLabel('bottom', 'Time Sweep')
    img_item = pg.ImageItem()
    plot.addItem(img_item)
    lut = pg.colormap.get('viridis').getLookupTable(0.0, 1.0, 256)
    img_item.setLookupTable(lut)
    img_item.setLevels([0, 1])  # Fixed amplitude scaling

    def update():
        global sweep_count, data_matrix
        new_sweep = get_radar_sweep()

        if sweep_count < max_chirps:
            data_matrix[:, sweep_count] = new_sweep
        else:
            data_matrix = np.roll(data_matrix, -1, axis=1)
            data_matrix[:, -1] = new_sweep

        sweep_count += 1

        if sweep_count >= chirps_per_refresh:
            img_item.setImage(data_matrix, autoLevels=False)

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(update_interval)

    QtWidgets.QApplication.instance().exec_()

if __name__ == '__main__':
    ProcessedMatrix()
