from PlutoSDR import PlutoSDR
import matplotlib.pyplot as plt

Pluto_IP = '192.168.2.1'
sample_rate = 60.5e6 # in hz
centerFrequency = 2.5e9 # in hz
tx_gain = -20 # in db
rx_gain = 40 # in db
rx_frame_duration = 1000 # in ms
rx_samples_per_frame = int((rx_frame_duration*(10**-3))/sample_rate)

chirp_type = "SawtoothWave" # Options: "SawtoothWave", "TriangularWave"
chirp_amplitude = (2**12)
chirp_bandwidth = 30e6 # hz
chirp_duration = 50 # ms

sdr_obj = PlutoSDR(Pluto_IP, sample_rate, centerFrequency, centerFrequency, rx_gain, tx_gain, rx_samples_per_frame)
sdr_obj.set_waveform(chirp_type, chirp_amplitude, chirp_bandwidth, chirp_duration)
sdr_obj.start_transmission()
received_data = self.receive_data()

plt.plot(np.arange(0, T + sample_period, sample_period), received_data)
plt.show()

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
