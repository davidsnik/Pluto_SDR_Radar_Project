import numpy as np
import matplotlib.pyplot as plt

# Simulated dechirped signal
np.random.seed(42)
N = 200
signal = np.abs(np.random.randn(N) * 2)
signal[50] += 15
signal[40] += 20

# Parameters
num_guard = 4
num_train = 5
PFA = 1e-2
total_cells = num_guard * 2 + num_train * 2 + 1

# Get sliding windows
from numpy.lib.stride_tricks import sliding_window_view

# Pad signal to handle edges
pad = num_guard + num_train
signal_padded = np.pad(signal, (pad, pad), mode='edge')
windows = sliding_window_view(signal_padded, total_cells)

# Create masks to exclude guard and CUT
mask = np.ones(total_cells, dtype=bool)
mask[num_train:num_train + 2*num_guard + 1] = False  # mask out guard cells + CUT

# Compute noise estimate using training cells
noise_level = np.sum(windows[:, mask], axis=1) / np.sum(mask)

# Calculate threshold scaling factor
alpha = num_train * (PFA**(-1/num_train) - 1)
thresholds = alpha * noise_level

# Detection
detections = signal > thresholds

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(signal, label='Signal')
plt.plot(detections * np.max(signal), 'r*', label='Detections')
plt.plot(thresholds, 'g--', label='Threshold')
plt.title("Vectorized CA-CFAR Detection")
plt.xlabel("Range Bin")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()
