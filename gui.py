from dataclasses import dataclass
from typing import Optional, Dict, Union, Callable

from ttkbootstrap import Variable, Frame, LabelFrame, StringVar, IntVar, Button, Canvas
from enum import Enum

import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PlutoSDR import PlutoSDR

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import numpy as np

# PADDING = 10
#
# DEFAULT_PLUTO_IP = '192.168.2.1'
# DEFAULT_SAMPLE_RATE = int(60.5e6)  # in hz
# DEFAULT_CENTRE_FREQUENCY = int(2.5e9)  # in hz
# DEFAULT_TX_GAIN = -20  # in db
# DEFAULT_RX_GAIN = 40  # in db
# DEFAULT_RX_FRAME_DURATION = 1  # in ms
#
# DEFAULT_CHIRP_AMPLITUDE = 2 ** 12
# DEFAULT_CHIRP_BANDWIDTH = int(30e6)
# DEFAULT_CHIRP_DURATION = 1  # in ms
# DEFAULT_PULSE_SLEEP_TIME = 1  # in ms
# DEFAULT_NUMBER_OF_COHERENT_PULSES = 10
#
# APP_TITLE = "Radar project"
#
# DEFAULT_VALUES = {"ip": DEFAULT_PLUTO_IP, "sample_rate": DEFAULT_SAMPLE_RATE,
#                   "centre_frequency": DEFAULT_CENTRE_FREQUENCY,
#                   "tx_gain": DEFAULT_TX_GAIN, "rx_gain": DEFAULT_RX_GAIN,
#                   "rx_frame_duration": DEFAULT_RX_FRAME_DURATION,
#                   "amplitude": DEFAULT_CHIRP_AMPLITUDE, "chirp_bandwidth": DEFAULT_CHIRP_BANDWIDTH,
#                   "chirp_duration": DEFAULT_CHIRP_DURATION, "pulse_sleep_time": DEFAULT_PULSE_SLEEP_TIME,
#                   "number_of_coherent_pulses": DEFAULT_NUMBER_OF_COHERENT_PULSES}


class WaveformTypes(Enum):
    Sawtooth = 1
    Triangular = 2


@dataclass
class TXWaveform:
    waveform_type_combobox_var: StringVar
    chirp_bandwidth_entry_var: IntVar
    chirp_duration_entry_var: IntVar
    amplitude_entry_var: IntVar
    number_of_coherent_pulses_entry_var: IntVar

    type: WaveformTypes
    current_chirp_bandwidth: int
    current_chirp_duration: int
    current_amplitude: int
    current_number_of_coherent_pulses: int


class App:
    def __init__(self, padding: int, app_title: str, default_entry_values: Dict[str, Union[int, str]]):
        self.sdr_object: Optional[PlutoSDR] = None
        self.centre_frequency: Optional[int] = None
        self.pulse_sleep_time: Optional[int] = None

        self.tx_waveform: Optional[TXWaveform] = None

        self.initial_configuration_frame: Optional[Frame] = None
        self.initial_configuration_label_frame: Optional[LabelFrame] = None
        self.initial_configuration_entries: Optional[Dict[str, Variable]] = None

        self.main_frame: Optional[Frame] = None
        self.waveform_config_label_frame: Optional[LabelFrame] = None
        self.tx_waveform_plot_widget: Optional[Canvas] = None
        self.transmit_button: Optional[Button] = None
        self.transmitting = False

        self.padding = padding
        self.app_title = app_title
        self.default_entry_values = default_entry_values

        self.root_window = ttk.Window(themename="superhero", title=app_title)
        self.root_window.resizable(False, False)

        self.style = self.root_window.style
        self.style.configure("TLabelframe.Label", font=("default", 22))
        self.style.configure("TLabel", font=("default", 18))

        self.title_frame = ttk.Frame(self.root_window, padding=self.padding)
        self.title_frame.pack(fill=BOTH)

        group_name_label = ttk.Label(self.title_frame, text="Group 4 - 2250", font=("default", 22, "bold"))
        group_name_label.pack(side=TOP, pady=self.padding)
        group_name_label_separator = ttk.Separator(self.title_frame, orient=HORIZONTAL)
        group_name_label_separator.pack(side=TOP, pady=0, fill=X, padx=self.padding)

        self.initial_configuration_frame, self.initial_configuration_label_frame, self.initial_configuration_entries = self.create_configuration_frame(
            self.root_window)
        self.initial_configuration_frame.pack()
        self.root_window.mainloop()

    def finish_configuration(self):
        if self.initial_configuration_frame is None or self.initial_configuration_label_frame is None or self.initial_configuration_entries is None:
            raise Exception("Could not finish configuration, as it was not initialised or has already been destroyed!")

        try:
            ip = self.initial_configuration_entries["ip"].get()
            sample_rate = self.initial_configuration_entries["sample_rate"].get()
            centre_frequency = self.initial_configuration_entries["centre_frequency"].get()
            rx_gain = self.initial_configuration_entries["rx_gain"].get()
            tx_gain = self.initial_configuration_entries["tx_gain"].get()
            rx_frame_duration = self.initial_configuration_entries["rx_frame_duration"].get()
            rx_samples_per_frame = int((rx_frame_duration * (10 ** -3)) / sample_rate)

            self.pulse_sleep_time = self.initial_configuration_entries["pulse_sleep_time"].get()

            self.centre_frequency = centre_frequency

            self.sdr_object = PlutoSDR(pluto_ip=ip, sample_rate=sample_rate, tx_center_freq=centre_frequency,
                                       rx_center_freq=centre_frequency, rx_gain=rx_gain, tx_gain=tx_gain,
                                       rx_samples_per_frame=rx_samples_per_frame)
        except Exception as error:
            print(error)
            self.initial_configuration_label_frame.configure(bootstyle=DANGER)
            return

        self.initial_configuration_frame.destroy()
        self.initial_configuration_frame = None
        self.initial_configuration_entries = None

        self.main_frame = self.create_main_frame(self.root_window)
        self.main_frame.pack()

    def create_configuration_frame(self, root: ttk.Window) -> (ttk.Frame, ttk.LabelFrame, Dict[str, ttk.Variable]):

        config_ip_var = ttk.StringVar(value=self.default_entry_values["ip"])
        config_sample_rate_var = ttk.IntVar(value=self.default_entry_values["sample_rate"])
        config_centre_frequency_var = ttk.IntVar(value=self.default_entry_values["centre_frequency"])
        config_tx_gain_var = ttk.IntVar(value=self.default_entry_values["tx_gain"])
        config_rx_gain_var = ttk.IntVar(value=self.default_entry_values["rx_gain"])
        config_rx_frame_duration_var = ttk.IntVar(value=self.default_entry_values["rx_frame_duration"])
        config_pulse_sleep_time = ttk.IntVar(value=self.default_entry_values["pulse_sleep_time"])

        entry_variables = {"ip": config_ip_var, "sample_rate": config_sample_rate_var,
                           "centre_frequency": config_centre_frequency_var,
                           "tx_gain": config_tx_gain_var, "rx_gain": config_rx_gain_var,
                           "rx_frame_duration": config_rx_frame_duration_var,
                           "pulse_sleep_time": config_pulse_sleep_time}

        configuration_frame = ttk.Frame(root, padding=self.padding)

        config_label_frame = ttk.LabelFrame(configuration_frame, text="Initial Configuration", padding=self.padding,
                                            bootstyle=PRIMARY)
        config_label_frame.pack(fill=BOTH)

        pluto_ip_label = ttk.Label(config_label_frame, text="Pluto IP", padding=self.padding)
        pluto_ip_label.grid(row=0, column=0)

        pluto_ip_entry = ttk.Entry(config_label_frame, textvariable=entry_variables["ip"])
        pluto_ip_entry.grid(row=0, column=1)

        pluto_sample_rate_label = ttk.Label(config_label_frame, text="Pluto Sample Rate", padding=self.padding)
        pluto_sample_rate_label.grid(row=1, column=0)

        pluto_sample_rate_entry = ttk.Entry(config_label_frame, textvariable=entry_variables["sample_rate"])
        pluto_sample_rate_entry.grid(row=1, column=1)

        pluto_centre_frequency_label = ttk.Label(config_label_frame, text="Centre Frequency", padding=self.padding)
        pluto_centre_frequency_label.grid(row=2, column=0)

        pluto_centre_frequency_entry = ttk.Entry(config_label_frame, textvariable=entry_variables["centre_frequency"])
        pluto_centre_frequency_entry.grid(row=2, column=1)

        pluto_tx_gain_label = ttk.Label(config_label_frame, text="TX Gain", padding=self.padding)
        pluto_tx_gain_label.grid(row=3, column=0)

        pluto_tx_gain_entry = ttk.Entry(config_label_frame, textvariable=entry_variables["tx_gain"])
        pluto_tx_gain_entry.grid(row=3, column=1)

        pluto_rx_gain_label = ttk.Label(config_label_frame, text="RX Gain", padding=self.padding)
        pluto_rx_gain_label.grid(row=4, column=0)

        pluto_rx_gain_entry = ttk.Entry(config_label_frame, textvariable=config_rx_gain_var)
        pluto_rx_gain_entry.grid(row=4, column=1)

        pluto_rx_frame_duration_label = ttk.Label(config_label_frame, text="RX Frame Length (ms)", padding=self.padding)
        pluto_rx_frame_duration_label.grid(row=5, column=0)

        pluto_rx_frame_duration_entry = ttk.Entry(config_label_frame, textvariable=entry_variables["rx_frame_duration"])
        pluto_rx_frame_duration_entry.grid(row=5, column=1)

        tx_pulse_sleep_time_label = ttk.Label(config_label_frame, text="Pulse Sleep Time (ms)", padding=self.padding)
        tx_pulse_sleep_time_label.grid(row=6, column=0)

        tx_pulse_sleep_time_entry = ttk.Entry(config_label_frame, textvariable=config_pulse_sleep_time)
        tx_pulse_sleep_time_entry.grid(row=6, column=1)

        complete_config_button = ttk.Button(config_label_frame, text="Complete", padding=self.padding, bootstyle=SUCCESS,
                                            command=self.finish_configuration)
        complete_config_button.grid(row=7, column=0, pady=self.padding, stick=EW, columnspan=2, padx=self.padding * 3)

        return configuration_frame, config_label_frame, entry_variables

    def update_waveform_parameters(self):
        if self.tx_waveform is None or self.centre_frequency is None or self.waveform_config_label_frame is None:
            raise Exception(
                "An update of the waveform parameters was requested, but the object has not yet been initialised!")

        try:
            self.tx_waveform.type = WaveformTypes[self.tx_waveform.waveform_type_combobox_var.get()]
            self.tx_waveform.current_chirp_bandwidth = self.tx_waveform.chirp_bandwidth_entry_var.get()
            self.tx_waveform.current_chirp_duration = self.tx_waveform.chirp_duration_entry_var.get()
            self.tx_waveform.current_amplitude = self.tx_waveform.amplitude_entry_var.get()
            self.tx_waveform.current_number_of_coherent_pulses = self.tx_waveform.number_of_coherent_pulses_entry_var.get()
            print(f"number: {self.tx_waveform.current_number_of_coherent_pulses}")
            self.sdr_object.set_waveform(chirp_type=self.tx_waveform.type,
                                         chirp_bandwidth=self.tx_waveform.current_chirp_bandwidth,
                                         chirp_duration=self.tx_waveform.current_chirp_duration,
                                         chirp_amplitude=self.tx_waveform.current_amplitude)
            self.waveform_config_label_frame.configure(bootstyle=INFO)

        except Exception as error:
            print(error)
            self.waveform_config_label_frame.configure(bootstyle=DANGER)

        figure = Figure(figsize=(4, 2), dpi=100)
        figure.tight_layout()
        axes = figure.add_subplot(1, 1, 1)

        time_axis = np.linspace(0, 3 * self.tx_waveform.current_chirp_duration, num=10000)
        chirp_function: Optional[Callable[[float], float]] = None

        match self.tx_waveform.type:
            case WaveformTypes.Sawtooth:
                chirp_function = lambda time: self.centre_frequency - self.tx_waveform.current_chirp_bandwidth / 2 + (
                            self.tx_waveform.current_chirp_bandwidth / self.tx_waveform.current_chirp_duration) * (
                                                          time % self.tx_waveform.current_chirp_duration)
            case WaveformTypes.Triangular:
                chirp_function = lambda time: self.centre_frequency - self.tx_waveform.current_chirp_bandwidth / 2 + (
                            2 * self.tx_waveform.current_chirp_bandwidth / self.tx_waveform.current_chirp_duration) * (
                                                          time % self.tx_waveform.current_chirp_duration) if (
                                                                                                                         time % self.tx_waveform.current_chirp_duration) < self.tx_waveform.current_chirp_duration / 2 else self.centre_frequency + self.tx_waveform.current_chirp_bandwidth / 2 - (
                            2 * self.tx_waveform.current_chirp_bandwidth / self.tx_waveform.current_chirp_duration) * ((
                                                                                                                                   time % self.tx_waveform.current_chirp_duration) - self.tx_waveform.current_chirp_duration / 2)
            case _:
                raise Exception("Unimplemented waveform type!")

        if self.tx_waveform_plot_widget is not None:
            self.tx_waveform_plot_widget.destroy()

        axes.plot(time_axis, [chirp_function(time) * 1e-9 for time in time_axis])
        axes.set_xlabel("Time (ms)")
        axes.set_ylabel("Frequency (GHz)")
        figure.tight_layout()

        canvas = FigureCanvasTkAgg(figure, master=self.waveform_config_label_frame)
        canvas.draw()
        self.tx_waveform_plot_widget = canvas.get_tk_widget()
        self.tx_waveform_plot_widget.grid(row=5, column=0, columnspan=2, sticky=EW, padx=self.padding,
                                          pady=self.padding)

    def transmit_button_pressed(self):
        if self.sdr_object is None or self.pulse_sleep_time is None:
            raise Exception("Tried to toggle transmission status but sdr object was never initialised!")

        self.transmit_button.configure(bootstyle=DANGER, text="Stop Transmitting")
        self.sdr_object.sliding_coherent_integration(1, self.tx_waveform.current_number_of_coherent_pulses, self.pulse_sleep_time)
        self.transmit_button.configure(bootstyle=SUCCESS, text="Start Transmitting")

        # if self.transmitting:
        #     self.transmit_button.configure(bootstyle=SUCCESS, text="Start Transmitting")
        #     self.sdr_object.stop_transmission()
        #     self.transmitting = False
        # else:
        #     self.transmit_button.configure(bootstyle=DANGER, text="Stop Transmitting")
        #     self.sdr_object.start_transmission()
        #     self.transmitting = True

    def user_input_for_waveform_parameters(self, _value1, _value2, _value3):
        self.waveform_config_label_frame.configure(bootstyle=WARNING)

    def create_main_frame(self, root: ttk.Window) -> ttk.Frame:

        main_frame = ttk.Frame(root, padding=self.padding)

        self.tx_waveform = TXWaveform
        self.tx_waveform.waveform_type_combobox_var = StringVar(value=next(iter(WaveformTypes)).name)
        self.tx_waveform.chirp_bandwidth_entry_var = IntVar(value=self.default_entry_values["chirp_bandwidth"])
        self.tx_waveform.chirp_duration_entry_var = IntVar(value=self.default_entry_values["chirp_duration"])
        self.tx_waveform.amplitude_entry_var = IntVar(value=self.default_entry_values["amplitude"])
        self.tx_waveform.number_of_coherent_pulses_entry_var = IntVar(value=self.default_entry_values["number_of_coherent_pulses"])

        self.tx_waveform.waveform_type_combobox_var.trace_add("write", self.user_input_for_waveform_parameters)
        self.tx_waveform.chirp_bandwidth_entry_var.trace_add("write", self.user_input_for_waveform_parameters)
        self.tx_waveform.chirp_duration_entry_var.trace_add("write", self.user_input_for_waveform_parameters)
        self.tx_waveform.amplitude_entry_var.trace_add("write", self.user_input_for_waveform_parameters)
        self.tx_waveform.number_of_coherent_pulses_entry_var.trace_add("write", self.user_input_for_waveform_parameters)

        self.tx_waveform.type = WaveformTypes[self.tx_waveform.waveform_type_combobox_var.get()]
        self.tx_waveform.current_chirp_bandwidth = self.tx_waveform.chirp_bandwidth_entry_var.get()
        self.tx_waveform.current_chirp_duration = self.tx_waveform.chirp_duration_entry_var.get()
        self.tx_waveform.current_amplitude = self.tx_waveform.amplitude_entry_var.get()
        self.tx_waveform.current_number_of_coherent_pulses = self.tx_waveform.number_of_coherent_pulses_entry_var.get()

        waveform_config_frame = ttk.Frame(main_frame, padding=self.padding)
        waveform_config_frame.grid(row=0, column=0)

        waveform_config_label_frame = ttk.LabelFrame(waveform_config_frame, text="TX Waveform", padding=self.padding,
                                                     bootstyle=INFO)
        waveform_config_label_frame.pack()
        self.waveform_config_label_frame = waveform_config_label_frame

        waveform_type_label = ttk.Label(self.waveform_config_label_frame, text="Chirp Type", padding=self.padding)
        waveform_type_label.grid(row=0, column=0)

        waveform_types = [waveform_type.name for waveform_type in WaveformTypes]

        waveform_type_combobox = ttk.Combobox(self.waveform_config_label_frame, values=waveform_types, state="readonly",
                                              textvariable=self.tx_waveform.waveform_type_combobox_var)
        waveform_type_combobox.grid(row=0, column=1)

        bandwidth_label = ttk.Label(self.waveform_config_label_frame, text="Chirp Bandwidth", padding=self.padding)
        bandwidth_label.grid(row=1, column=0)

        bandwidth_entry = ttk.Entry(self.waveform_config_label_frame,
                                    textvariable=self.tx_waveform.chirp_bandwidth_entry_var)
        bandwidth_entry.grid(row=1, column=1)

        duration_label = ttk.Label(self.waveform_config_label_frame, text="Chirp Duration (ms)", padding=self.padding)
        duration_label.grid(row=2, column=0)

        duration_entry = ttk.Entry(self.waveform_config_label_frame,
                                   textvariable=self.tx_waveform.chirp_duration_entry_var)
        duration_entry.grid(row=2, column=1)

        amplitude_label = ttk.Label(self.waveform_config_label_frame, text="Amplitude", padding=self.padding)
        amplitude_label.grid(row=3, column=0)

        amplitude_entry = ttk.Entry(self.waveform_config_label_frame, textvariable=self.tx_waveform.amplitude_entry_var)
        amplitude_entry.grid(row=3, column=1)

        number_of_coherent_pulses_label = ttk.Label(self.waveform_config_label_frame, text="Number of coherent pulses", padding=self.padding)
        number_of_coherent_pulses_label.grid(row=4, column=0)

        number_of_coherent_pulses_entry = ttk.Entry(self.waveform_config_label_frame, textvariable=self.tx_waveform.number_of_coherent_pulses_entry_var)
        number_of_coherent_pulses_entry.grid(row=4, column=1)


        apply_button = ttk.Button(self.waveform_config_label_frame, text="Apply", padding=self.padding, bootstyle=INFO,
                                  command=self.update_waveform_parameters)
        apply_button.grid(row=5, column=0)

        transmit_button = ttk.Button(self.waveform_config_label_frame, text="Start Transmitting", padding=self.padding,
                                     bootstyle=SUCCESS, command=self.transmit_button_pressed)
        transmit_button.grid(row=5, column=1)

        self.transmit_button = transmit_button



        return main_frame




# app = App(PADDING, APP_TITLE, DEFAULT_VALUES)