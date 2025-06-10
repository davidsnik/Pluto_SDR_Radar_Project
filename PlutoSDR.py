import adi
import numpy as np
import scipy.fft as fft
import time

class PlutoSDR:
    def __init__(self, PlutoIP, sample_rate, tx_center_freq, rx_center_freq, rx_gain, tx_gain, rx_samples_per_frame, skip_pluto_configuration=False):
        if not skip_pluto_configuration:
            PlutoIP = 'ip:'+PlutoIP
            my_sdr = adi.Pluto(uri=PlutoIP)
            my_sdr.sample_rate = int(sample_rate)
            my_sdr.tx_rf_bandwidth = int(sample_rate)
            my_sdr.rx_rf_bandwidth = int(sample_rate)
            my_sdr.rx_lo = int(rx_center_freq)
            my_sdr.tx_lo = int(tx_center_freq)
            my_sdr.rx_enabled_channels = [0]
            sample_rate = int(my_sdr.sample_rate)
            my_sdr.gain_control_mode_chan0 = "manual"
            my_sdr.rx_hardwaregain_chan0 = int(rx_gain)
            my_sdr._rxadc.set_kernel_buffers_count(1)

            my_sdr.tx_enabled_channels = [0]
            my_sdr.tx_hardwaregain_chan0 = int(tx_gain)

            frame_length_samples = rx_samples_per_frame

            N_rx = int(1 * frame_length_samples)
            my_sdr.rx_buffer_size = N_rx

            self.pluto_interface = my_sdr
        else:
            self.pluto_interface = None

        self.Pluto_IP = PlutoIP
        self.sample_rate = sample_rate
        self.tx_center_freq = tx_center_freq
        self.rx_center_freq = rx_center_freq
        self.rx_gain = rx_gain
        self.tx_gain = tx_gain
        self.rx_samples_per_frame = rx_samples_per_frame

        self.chirp_time_axis = None
        self.chirp_type = None
        self.chirp_amplitude = None
        self.chirp_bandwidth = None
        self.chirp_duration_ms = None
        self.tx_iq = None
        self.rx_iq = None
        self.s_beat = None
        self.f_beat = None
        self.s_beat_buffer = []
        self.s_beat_signal_coherent_integration = None
        self.s_beat_freq_coherent_integration = None

    def set_waveform(self, chirp_type, chirp_amplitude, chirp_bandwidth, chirp_duration_ms):
        sample_period = 1 / self.sample_rate
        duration_seconds = chirp_duration_ms * 1e-3  # Convert ms to seconds

        time = np.arange(0, duration_seconds, sample_period)

        window = np.hanning(len(time))

        match chirp_type:
            case "SawtoothWave":
                chirp_slope = chirp_bandwidth / duration_seconds  # units: Hz/s
                phase = 2 * np.pi * (0.5 * chirp_slope * (time**2))
                self.tx_iq = chirp_amplitude * np.exp(1j * phase) * window

            case "TriangularWave":
                half = len(time) // 2
                t_up = time[:half]
                t_down = time[half:]

                chirp_slope = chirp_bandwidth / (duration_seconds / 2)  # Hz/s for each half

                # Up-chirp: 0 -> bandwidth
                phase_up = 2 * np.pi * (0.5 * chirp_slope * t_up**2)
                # Down-chirp: bandwidth -> 0, ensure phase continuity
                tau = t_down - t_up[-1]
                phase_down = phase_up[-1] + 2 * np.pi * (
                    chirp_bandwidth * tau - 0.5 * chirp_slope * tau**2
                )

                self.tx_iq = chirp_amplitude * np.concatenate([
                    np.exp(1j * phase_up),
                    np.exp(1j * phase_down)
                ]) * window

            case _:
                return None

        self.chirp_time_axis = time
        self.chirp_type = chirp_type
        self.chirp_amplitude = chirp_amplitude
        self.chirp_bandwidth = chirp_bandwidth  # in Hz
        self.chirp_duration_ms = chirp_duration_ms  # in ms


    def start_transmission(self):
        if self.chirp_type != None:
            self.pluto_interface._rx_init_channels()
            self.pluto_interface.tx_cyclic_buffer = True  # must be true to use the continuos transmission
            self.pluto_interface.tx(self.tx_iq)
        else:
            print("Start transmission not working since waveform config unset")

    def receive_data(self):
        if self.chirp_type != None:
            self.rx_iq = self.pluto_interface.rx()
            N_min = min(len(self.rx_iq), len(self.tx_iq))
            self.s_beat = self.rx_iq[:N_min] * np.conj(self.tx_iq[:N_min])
            return self.s_beat
        else:
            print("Receive data not working since waveform config unset")
    def stop_transmission(self):
        if self.chirp_type != None:
            self.pluto_interface.tx_destroy_buffer()
        else:
            print("Stop transmission not working since waveform config unset")

    def get_beat_signal(self):
        if self.chirp_type != None:
            self.start_transmission()
            s_beat = self.receive_data()
            self.stop_transmission()

            return s_beat
        else:
            print("Get beat freq not possible since waveform not set")

    def sliding_coherent_integration(self, number_of_pulses_to_add, pulse_buffer_size, pulse_sleep_time_ms):
        pulses_to_add = []
        for i in range(number_of_pulses_to_add):
            time.sleep((pulse_sleep_time_ms*(10**-3))/2)
            pulses_to_add.append(self.get_beat_signal())
            time.sleep((pulse_sleep_time_ms*(10**-3))/2)

        if len(self.s_beat_buffer) >= pulse_buffer_size:
            self.s_beat_buffer = self.s_beat_buffer[pulse_buffer_size-number_of_pulses_to_add+(len(self.s_beat_buffer)-pulse_buffer_size):]
        self.s_beat_buffer.extend(pulses_to_add)

        min_len = min([len(s_beat_sig) for s_beat_sig in self.s_beat_buffer])

        s_beat_signal_coherent_integration = np.mean([s_beat_sig[:min_len] for s_beat_sig in self.s_beat_buffer])
        self.s_beat_signal_coherent_integration = s_beat_signal_coherent_integration

        spectrum = fft.fft(s_beat_signal_coherent_integration)
        magnitude = np.abs(spectrum)
        peak_index = np.argmax(magnitude)
        frequencies = fft.fftfreq(len(s_beat_signal_coherent_integration), d=1/self.sample_rate)
        s_beat_freq_coherent_integration = np.abs(frequencies[peak_index])
        self.s_beat_freq_coherent_integration = s_beat_freq_coherent_integration

        return s_beat_signal_coherent_integration, s_beat_freq_coherent_integration
