import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile

# Load mixed sounds file
filename = 'himna.wav'
fs, y = wavfile.read(filename)

# Define filter parameters
f0 = [1500.0, 2000.0, 2750.0, 3250.0]
Q = [150.0, 150.0, 150.0, 150.0]
fc = 2500.0

# Create notch filters
b = []
a = []
for i in range(len(f0)):
    w0 = f0[i] / (fs / 2)
    b_i, a_i = signal.iirnotch(w0, Q[i])
    b.append(b_i)
    a.append(a_i)

# Create low-pass filter
order = 4
nyquist = 0.5 * fs
fc_norm = fc / nyquist
b_lp, a_lp = signal.butter(order, fc_norm, btype='lowpass')

# Apply filters
y_filtered = y.copy()
for i in range(len(f0)):
    y_filtered = signal.filtfilt(b[i], a[i], y_filtered)
y_filtered = signal.filtfilt(b_lp, a_lp, y_filtered)

# Save filtered audio
wavfile.write('filtered_mixed_sounds.wav', fs, y_filtered)


