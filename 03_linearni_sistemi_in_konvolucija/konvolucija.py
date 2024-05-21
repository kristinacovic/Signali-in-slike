import numpy as np
import matplotlib.pyplot as plt

def konvolucija_offline(signal: np.ndarray, impulz: np.ndarray, dolzina: str) -> np.ndarray:
    N = signal.shape[0]     # length of signal
    K = impulz.shape[0]     # length of impulse response
    C = 1 if len(signal.shape) == 1 else signal.shape[1]    # number of channels

    # checking if the input signal has multiplee channels or just one
    # if it has multiple channels, it preforms convolution on each channel separately
    # if it has only one channel, then it preforms convolution on the single channel
    if C > 1:
        # Convolution for multiple channels
        if len(impulz.shape) == 1:  # checking if it is 1d, if it is 1d, reshaping it into 2d array, with single column
            impulz = impulz[:, np.newaxis]    

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

if __name__ == '__main__':
    # 20 samples
    signal = np.array([1, 0, -1, 2, 4, 1, 0, -1, 2, 4, 1, 0, -1, 2, 4, 1, 0, -1, 2, 4])
    # 5 samples
    impulz = np.array([1, -1, 2, 1, 3])

    # calculating convolution
    konv = konvolucija_offline(signal, impulz, 'polna')

    # plotting signal, impulz and rezultat
    fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

    axs[0].plot(signal)
    axs[0].set_title('Signal')
    axs[0].set_ylabel('Amplituda')

    axs[1].stem(impulz)
    axs[1].set_title('Impulz')
    axs[1].set_ylabel('Amplituda')

    axs[2].plot(konv)
    axs[2].set_title('Konvolucija')
    axs[2].set_ylabel('Amplituda')
    axs[2].set_xlabel('Vzorec')

    plt.tight_layout()
    plt.show()
