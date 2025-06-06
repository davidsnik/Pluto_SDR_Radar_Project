import adi
import numpy as np
import typing as t

class PlutoSDR:
    def __init__(self, PlutoIP, sample_rate, center_freq, rx_gain, tx_gain, rx_SamplesPerFrame):
        PlutoIP = 'ip:'+PlutoIP
        my_sdr = adi.Pluto(uri=PlutoIP)
        my_sdr.sample_rate = int(sample_rate)
        my_sdr.tx_rf_bandwidth = int(sample_rate)
        my_sdr.rx_rf_bandwidth = int(sample_rate)
        my_sdr.rx_lo = int(center_freq)
        my_sdr.tx_lo = int(center_freq)
        my_sdr.rx_enabled_channels = [0]
        sample_rate = int(sample_rate)
        my_sdr.gain_control_mode_chan0 = "manual"
        my_sdr.rx_hardwaregain_chan0 = int(rx_gain)
        my_sdr._rxadc.set_kernel_buffers_count(1)
        my_sdr.tx_enabled_channels = [0]
        my_sdr.tx_hardwaregain_chan0 = int(tx_gain)
        frame_length_samples = rx_SamplesPerFrame
    
        N_rx = int(1 * frame_length_samples)
        my_sdr.rx_buffer_size = N_rx

        self.PlutoIP = PlutoIP
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.rx_gain = rx_gain
        self.tx_gain = tx_gain
        self.rx_SamplesPerFrame = rx_SamplesPerFrame
        self.pluto_interface = my_sdr
    
    def set_waveform(self, chirp_type, bandwidth, chirp_duration, carrier_frequency):
        sample_period = 1/self.sample_rate

        time = np.arange(0, T + sample_period, sample_period)

        match chirp_type:
            case "SawtoothWave":
                chirp_slope = bandwidth/chirp_duration
                self.iq = (2**12) * np.exp(1j*np.pi*chirp_slope*(time**2))
            case "TriangularWave":
                pass # TODO

        self.current_waveform_type = waveform_type
        self.iq = iq

    def start_transmission(self):
        self.pluto_interface.tx_cyclic_buffer = True  # must be true to use the continuos transmission
        self.pluto_interface.tx(self.iq)

    def receive_data(self, frame_length_samples):
        self.pluto_interface._rx_init_channels()
        received_array = self.pluto_interface.rx()     
        return received_array.tolist()

    def stop_transmission(self):
        self.pluto_interface.tx_destroy_buffer()



# %%%%%%%%%%%%%%%%%%%%%%%%% TX Waveform generation %%%%%%%%%%%%%%%%%%%%%%%%%%
# B = 30e6;       % Chirp bandwidth (Hz)
# T = 100e-6;       % Chirp duration (s)
# f0 = 2.5e9;      % Carrier frequency of the RF signal (Hz)
# fs = PlutoSamprate;     % Sampling rate of all the signals in this simulation (Hz) 
# c = 3e8;        % Speed of light (m/s)


# chirp_duration = 1; % Chirp duration (ms)
# % t = 1/fs:1/fs:chirp_duration*1e-3;  
# t = single(0:1/fs:(chirp_duration*1e-3 ));  

# % t = 0:1/fs:(T-1/fs); 
# k = B / T;     % Chirp slope defined as ratio of bandwidth over duration
# sig_A = exp(1j*2*pi*(0.5*k*t.^2));

# tx_waveform = 2^12.*sig_A.';  % # The PlutoSDR expects samples to be between 
# %                           -2^14 and +2^14, not -1 and +1 like some SDRs
