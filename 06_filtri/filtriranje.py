import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def odstrani_kvadratni_signal(s: np.ndarray, f: float, fvz: int) -> np.ndarray:
    # Define the frequency range to filter
    f_range = [f * i for i in range(1, int(fvz / (2 * f)) + 1, 2)]

    # Determine the width of each frequency band
    bandwidth = f / 2

    # Define the filter order
    order = 4

    # Create the filter bank
    filters = [signal.butter(order, [fr - bandwidth, fr + bandwidth], btype='bandstop', fs=fvz) for fr in f_range]

    # Apply each filter in sequence
    for f in filters:
        s = signal.filtfilt(f[0], f[1], s)

    # Return the filtered signal
    return s



if __name__ == '__main__':
    fvz = 200
    t = np.arange(0, 5, 1/fvz)
    s_kvadratni = signal.square(2 * np.pi * 10 * t)
    s_zagast = signal.sawtooth(2 * np.pi * 15 * t)
    s_vhodni = s_kvadratni + s_zagast

    s_filtriran = odstrani_kvadratni_signal(s_vhodni, 10, fvz)

    fig, axs = plt.subplots(4, 1, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.5)

    axs[0].plot(t, s_kvadratni)
    axs[0].set_title('Kvadratni signal')
    axs[0].set_xlabel('Čas [s]')
    axs[0].set_ylabel('Amplituda')

    axs[1].plot(t, s_zagast)
    axs[1].set_title('Žagast signal')
    axs[1].set_xlabel('Čas [s]')
    axs[1].set_ylabel('Amplituda')

    axs[2].plot(t, s_vhodni)
    axs[2].set_title('Vhodni signal')
    axs[2].set_xlabel('Čas [s]')
    axs[2].set_ylabel('Amplituda')

    axs[3].plot(t, s_zagast, label='Žagast signal')
    axs[3].plot(t, s_filtriran, label='Filtriran signal')
    axs[3].set_title('Filtriranje')
    axs[3].set_xlabel('Čas [s]')
    axs[3].set_ylabel('Amplituda')
    axs[3].legend()

    plt.show()
