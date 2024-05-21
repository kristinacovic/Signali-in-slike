import numpy as np
import scipy.io.wavfile as wav


def konvolucija_offline(signal: np.ndarray, impulz: np.ndarray, dolzina: str) -> np.ndarray:
    N = signal.shape[0]     # length of signal
    K = impulz.shape[0]     # length of impulse response
    C = 1 if len(signal.shape) == 1 else signal.shape[1]

    # checking if the input signal has multiplee channels or just one
    # if it has multiple channels, it preforms convolution on each channel separately
    # if it has only one channel, then it preforms convolution on the single channel
    if C > 1:
        # Convolution for multiple channels
        if len(impulz.shape) == 1:  # checking if it is 1d, if it is 1d, reshaping it into 2d array, with single column
            impulz = impulz[:, np.newaxis]     # done so convolution works 

        # if dolzina is full, out has the shape of 'N+K-1', and if it is not full, then it has shape of 'N-K+1'
        M = N + K - 1 if dolzina == 'polna' else N - K + 1
        out = np.zeros((M, C))

        # Iterate over the number of channels in the input signal
        for i in range(C):      
            for j in range(K-1, N):     # K-1 TO N
                out_idx = j-K+1 if dolzina == 'veljavna' else j     # if dolzina == veljavna, out_idx j-K+1 , else it is j
                # convolution between input signal and impulse response
                # out_idx is the starting index of the output
                # j-K+1 is used if dolzina is 'veljavna', otherwise j is used
                # np.sum calculates the dot product between the signal and the reversed impulse response
                out[out_idx, i] = np.sum(signal[j-K+1:j+1, i] * impulz[::-1, 0])

    else:
        # Convolution for one channel

        # If the impulz is 2D, flatten it to 1D
        if len(impulz.shape) == 2:
            impulz = impulz.flatten()

        # Calculate the output size based on the mode (polna or veljavna)
        M = N + K - 1 if dolzina == 'polna' else N - K + 1
        out = np.zeros((M,))

        # Iterate over each sample in the signal and perform convolution
        for j in range(K-1, N):
            out_idx = j-K+1 if dolzina == 'veljavna' else j      # Calculate the output index based on the mode
            out[out_idx] = np.sum(signal[j-K+1:j+1] * impulz[::-1])     # Calculate the convolution sum for the current sample
    return out

# Load the impulse response files
impulse1 = wav.read('efekt1.wav')[1]
impulse2 = wav.read('efekt2.wav')[1]
impulse3 = wav.read('efekt3.wav')[1]

# Load the audio signal
fs, signal = wav.read('signal.wav')

#signal = np.mean(signal, axis=1)

# Apply convolution using the first impulse response
processed_signal1 = konvolucija_offline(signal, impulse1, 'polna')

# Apply convolution using the second impulse response
processed_signal2 = konvolucija_offline(signal, impulse2, 'polna')

# Apply convolution using the third impulse response
processed_signal3 = konvolucija_offline(signal, impulse3, 'polna')

# Save the processed signals
wav.write('rezultat1.wav', fs, processed_signal1.astype(np.int16))
wav.write('rezultat2.wav', fs, processed_signal2.astype(np.int16))
wav.write('rezultat3.wav', fs, processed_signal3.astype(np.int16))