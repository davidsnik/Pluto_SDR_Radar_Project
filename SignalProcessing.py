import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy.signal import windows
from scipy.fft import fft, fftshift, fftfreq


import sys
from main import sample_rate, chirp_bandwidth, chirp_duration, centerFrequency
# chirp_bandwidth = 30e6 # hz
# chirp_duration = 0.000128 # ms
# sample_rate = 1e6# in hz
# centerFrequency = 2.5e9 # in hz
max_chirps = 255
chirps_per_refresh = 1


c = 3e8
# update_interval = int(chirps_per_refresh*chirp_duration*1e3)
update_interval = 1

def next_pow2(n):
    return 1 if n == 0 else 2**int(np.ceil(np.log2(n)))

def DoubleFFT():
    
    global data_matrix, sweep_count
    from processingtest import RadarChirpSimulator
    sim = RadarChirpSimulator()
    
    rows = int(chirp_duration*sample_rate)  # samples per chirp (fast time)
    cols = max_chirps                     # number of chirps (slow time)
    data_matrix = np.zeros((rows, cols), dtype= np.complex64)
    k = chirp_bandwidth / chirp_duration
    sweep_count = 0
    
    N_FFT = next_pow2(rows)
    half_idx = N_FFT // 2
    
    f_axis = fftshift(fftfreq(N_FFT, d=1/sample_rate))
    f_axis_pos = f_axis[half_idx:] 

    range_axis = (c * f_axis_pos) / (2 * k)
    
    
    app = QtWidgets.QApplication(sys.argv)
    lut = pg.colormap.get('viridis').getLookupTable(0.0, 1.0, 256)
    
    #First window
    win1 = pg.GraphicsLayoutWidget(title="Range-Time Plot")
    plot1 = win1.addPlot()
    plot1.setLabel('left', 'Time (s)')
    plot1.setLabel('bottom', 'Range (m)')
    win1.show()
    img_item = pg.ImageItem()
    plot1.addItem(img_item)
    img_item.setLookupTable(lut)
    img_item.setLevels([0, 1])
    
    # Second window
    win2 = pg.GraphicsLayoutWidget(title="Doppler-Range Plot")
    plot2 = win2.addPlot()
    plot2.setLabel('left', 'Velocity (m/s)')
    plot2.setLabel('bottom', 'Range (m)')
    win2.show()
    img_doppler = pg.ImageItem()
    plot2.addItem(img_doppler)
    img_doppler.setLookupTable(lut)
    img_doppler.setLevels([0,1])
    
    def update():
        global data_matrix, sweep_count
        new_sweep = sim.get_next_chirp()

        elapsed_time = sweep_count * chirp_duration
        window_width = max_chirps * chirp_duration
        start_time = max(0, elapsed_time - window_width)
        
        if sweep_count < max_chirps:
            data_matrix[:, sweep_count] = new_sweep
        else:
            data_matrix = np.roll(data_matrix, -1, axis=1)
            data_matrix[:, -1] = new_sweep
        sweep_count += 1

        # FFT along fast time (range)
        FFT_data = fftshift(fft(data_matrix, n=N_FFT, axis=0), axes=0)[half_idx:, :]
        norm_fft = np.abs(FFT_data)/np.max(np.abs(FFT_data))

        plot1.setYRange(start_time, start_time + window_width, padding=0)
        
        img_item.setImage(norm_fft, autoLevels=False)
        img_item.setRect(QtCore.QRectF(range_axis[0], start_time,
                                    range_axis[-1] - range_axis[0], window_width))

        # Determines on how many chirps the fft will be applied.
        N_Doppler = max_chirps
        N_FFT_doppler = next_pow2(N_Doppler)
        
        if sweep_count >= N_Doppler:
            #This ifelse maks sure slow_time_profiles always chooses something between 0 and 255, since FFT_pos has shape (..,255)
            if sweep_count<=cols:
                slow_time_profiles = FFT_data[:, sweep_count-N_Doppler : sweep_count] #Choose last N_doppler chirps to apply FFT on
            else:
                slow_time_profiles = FFT_data[:, 255-N_Doppler : 255]
            
            doppler_fft = fftshift(fft(slow_time_profiles,n = N_FFT_doppler ,axis=1), axes=1)
            doppler_map = np.abs(doppler_fft)/np.max(np.abs(doppler_fft))
            
            f_axis_dop = fftshift(fftfreq(N_FFT_doppler, d=chirp_duration))
            vel_axis = (c * f_axis_dop) / (2 * centerFrequency)

            img_doppler.setImage(doppler_map, autoLevels=False)
            img_doppler.setRect(QtCore.QRectF(range_axis[0], vel_axis[0],
                                        range_axis[-1] - range_axis[0],
                                        vel_axis[-1] - vel_axis[0]))
            plot2.setLabel('bottom', 'Range [m]')
            plot2.setLabel('left', 'Velocity [m/s]')
            return norm_fft, doppler_map
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(update_interval)

    QtWidgets.QApplication.instance().exec()

if __name__ == '__main__':
    DoubleFFT()