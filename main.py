from PlutoSDR import PlutoSDR
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft

from gui import App

PADDING = 10

DEFAULT_PLUTO_IP = '192.168.2.1'
DEFAULT_SAMPLE_RATE = int(60.5e6)  # in hz
DEFAULT_CENTRE_FREQUENCY = int(2.5e9)  # in hz
DEFAULT_TX_GAIN = -20  # in db
DEFAULT_RX_GAIN = 40  # in db
DEFAULT_RX_FRAME_DURATION = 1  # in ms

DEFAULT_CHIRP_AMPLITUDE = 2 ** 12
DEFAULT_CHIRP_BANDWIDTH = int(30e6)
DEFAULT_CHIRP_DURATION = 1  # in ms
DEFAULT_PULSE_SLEEP_TIME = 1  # in ms
DEFAULT_NUMBER_OF_COHERENT_PULSES = 10

APP_TITLE = "Radar project"

DEFAULT_VALUES = {"ip": DEFAULT_PLUTO_IP, "sample_rate": DEFAULT_SAMPLE_RATE,
                  "centre_frequency": DEFAULT_CENTRE_FREQUENCY,
                  "tx_gain": DEFAULT_TX_GAIN, "rx_gain": DEFAULT_RX_GAIN,
                  "rx_frame_duration": DEFAULT_RX_FRAME_DURATION,
                  "amplitude": DEFAULT_CHIRP_AMPLITUDE, "chirp_bandwidth": DEFAULT_CHIRP_BANDWIDTH,
                  "chirp_duration": DEFAULT_CHIRP_DURATION, "pulse_sleep_time": DEFAULT_PULSE_SLEEP_TIME,
                  "number_of_coherent_pulses": DEFAULT_NUMBER_OF_COHERENT_PULSES}

app = App(PADDING, APP_TITLE, DEFAULT_VALUES)

# sdr_obj = PlutoSDR(PlutoIP, sample_rate, centerFrequency, centerFrequency, rx_gain, tx_gain, rx_samples_per_frame, skip_pluto_configuration=True)
# sdr_obj.set_waveform(chirp_type, chirp_amplitude, chirp_bandwidth, chirp_duration)
#sdr_obj.start_transmission()
#received_data = sdr_obj.receive_data()

# f, t, Z = stft(sdr_obj.tx_iq, fs = sample_rate, nperseg = 256, return_onesided = True)
# # pos_freq = f>=0
# # f_pos = f[pos_freq]
# # Z_pos = Z[pos_freq]
# plt.figure(figsize = (10,4))
# plt.pcolormesh(t,f,np.abs(Z)/np.max(np.abs(Z)), shading = 'gouraud')
# plt.tight_layout()
# plt.show()

# %%%%%%%%%%%%%%%%%%%% Pluto's parameters configuration %%%%%%%%%%%%%%%%%%%%%

# % If you want repeatable alignment between transmit and receive then the 
# % rx_lo, tx_lo and sample_rate can only be set once after power up. If you
# % want to change the following parameters' values after the first run you 
# % must reboot the Pluto (disconnect and reconnect it)

# Pluto_IP = '192.168.2.1';
# PlutoSamprate = 60.5e6;  % Sampling frequency (Hz): Pluto can sample up to
# %                       61 MHz but due to the USB 2.0 interface you have to
# %                       choose values lower or equal to 5MHz if you want to
# %                       receive 100% of samples over time.
# centerFrequency = 2.5e9;  % Pluto operating frequency (Hz) must be between
# %                         70MHz and 6GHz
# tx_gain = -20; %-45  % Pluto TX channel Gain must be between 0 and -88
# rx_gain = 40;% 0   % Pluto RX channel Gain must be between -3 and 70

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
