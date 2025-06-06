import adi
import numpy as np

class PlutoSDR:
    def __init__(PlutoIP, tx_buffer_size, sample_rate, tx_center_freq, rx_center_freq, rx_gain, tx_gain, rx_samples_per_frame):
        PlutoIP = 'ip:'+PlutoIP
        my_sdr = adi.Pluto(uri=PlutoIP)
        my_sdr.sample_rate = int(sample_rate)
        my_sdr.tx_rf_bandwidth = int(sample_rate)
        my_sdr.rx_rf_bandwidth = int(sample_rate)
        my_sdr.rx_lo = int(tx_center_freq)
        my_sdr.tx_lo = int(rx_center_freq)
        my_sdr.rx_output_type    
        my_sdr.rx_enabled_channels = [0]
        sample_rate = int(my_sdr.sample_rate)
        my_sdr.gain_control_mode_chan0 = "manual"
        my_sdr.rx_hardwaregain_chan0 = int(rx_gain)
        my_sdr._rxadc.set_kernel_buffers_count(1)

        my_sdr.tx_enabled_channels = [0]
        my_sdr.tx_hardwaregain_chan0 = int(tx_gain)
        
        frame_length_samples = rx_samples_per_frame
        
        if frame_length_samples != tx_buffer_size:
            frame_length_samples = int(tx_buffer_size)
        
        N_rx = int(1 * frame_length_samples)
        my_sdr.rx_buffer_size = N_rx

        self.Pluto_IP = PlutoIP
        self.tx_buffer_size = tx_buffer_size
        self.sample_rate = sample_rate
        self.tx_center_freq = tx_center_freq
        self.rx_center_freq = rx_center_freq
        self.rx_gain = rx_gain
        self.tx_gain = tx_gain
        self.rx_samples_per_frame = rx_samples_per_frame

        self.chirp_type = None
        self.chirp_amplitude = None
        self.chirp_bandwidth = None
        self.chirp_duration = None
        self.iq = None

        self.pluto_interface = my_sdr
    
    def set_waveform(self, chirp_type, chirp_amplitude, chirp_bandwidth, chirp_duration):
        sample_period = 1/self.sample_rate

        time = np.arange(0, T + sample_period, sample_period)

        match chirp_type:
            case "SawtoothWave":
                chirp_slope = chirp_bandwidth/(chirp_duration*(10**-3))
                self.iq = chirp_amplitude * np.exp(1j*np.pi*chirp_slope*(time**2))
            case "TriangularWave":
                pass # TODO

        self.chirp_type = chirp_type
        self.chirp_amplitude = chirp_amplitude
        self.chirp_bandwidth = chirp_bandwidth
        self.chirp_duration = chirp_duration

    def start_transmission(self):
        if self.iq != None:
            self.pluto_interface._rx_init_channels()
            self.pluto_interface.tx_cyclic_buffer = True  # must be true to use the continuos transmission
            self.pluto_interface.tx(self.iq)
        else:
            print("Start transmission not working since waveform config unset")

    def receive_data(self):
        if self.iq != None:
            received_array = self.pluto_interface.rx()     
            return received_array.tolist()
        else:
            print("Receive data not working since waveform config unset")
    def stop_transmission(self):
        if self.iq != None:
            self.pluto_interface.tx_destroy_buffer()
        else:
            print("Stop transmission not working since waveform config unset")