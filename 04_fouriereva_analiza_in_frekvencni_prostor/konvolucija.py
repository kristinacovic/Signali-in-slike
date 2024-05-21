import numpy as np
import matplotlib as plt

def konvolucija_fft(signal: np.ndarray, impulz: np.ndarray, rob: str) -> np.ndarray:
    # checking dimensions of signal and impulz
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)  # converts 1d signal to 2d
    if impulz.ndim == 1:
        impulz = impulz.reshape(-1, 1)  # converts 1d impulz to 2d
    
    # determines singal and impulz length
    N, C = signal.shape # N - number of samples of signal, C - number of channels
    K, _ = impulz.shape # K - number of samples of impulz
    
    # max length between length of signal and length of impulse, determines the size of padding required for convolution
    L = max(N, K)   
    # padding both arrays with zeros

    if rob == 'ničle':
        # ((0, L-N), (0, 0)) -  amount of padding to be added
        signal = np.pad(signal, ((0, L-N), (0, 0)), mode='constant')    # 'constant' means that it has to be done with zeros
        impulz = np.pad(impulz, ((0, L-K), (0, 0)), mode='constant')

    # padding both arrays with mirrored values
    elif rob == 'zrcaljen':
        # 'reflect' specifies that the padding should be done with values mirrored from the edges
        signal = np.pad(signal, ((K-1, K-1), (0, 0)), mode='reflect') 
        impulz = np.pad(impulz, ((K-1, K-1), (0, 0)), mode='reflect')
        # further padded with zeros, same as at 'nicle'
        signal = np.pad(signal, ((0, L-N), (0, 0)), mode='constant')
        impulz = np.pad(impulz, ((0, L-K), (0, 0)), mode='constant')
    elif rob == 'krožni':
        #'wrap' - specifies that the padding is done with values wrapped from the edges.
        signal = np.pad(signal, ((0, L-N), (0, 0)), mode='wrap')
        impulz = np.pad(impulz, ((0, L-K), (0, 0)), mode='wrap')
    
    signal_fft = np.fft.fft(signal, axis=0) # signal fft
    impulz_fft = np.fft.fft(impulz, axis=0) # impulz fft
    konvolucija_fft = signal_fft * impulz_fft   # calculating convoultion of signal and impulz, done in frequency domain
    konvolucija = np.real(np.fft.ifft(konvolucija_fft, axis=0)[:N])  # np.real() - because ifft produces small imaginary parts
    
    return konvolucija

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    
    signal = np.array([1,2,3,2,4,1,5,4,2,3])
    impulz = np.array([-1, 0, 1])
    
    konv_nicle = konvolucija_fft(signal, impulz, 'ničle')
    konv_zrcaljen = konvolucija_fft(signal, impulz, 'zrcaljen')
    konv_krozni = konvolucija_fft(signal, impulz, 'krožni')
    
    fig, axs = plt.subplots(5, 1, figsize=(10, 10))

    axs[0].plot(signal, 'b*-')
    axs[0].set_title('Vhodni signal')
    axs[0].set_xlabel('Čas [vzorci]')
    axs[0].set_ylabel('Amplituda')

    axs[1].plot(impulz, 'b*-')
    axs[1].set_title('Impulz')
    axs[1].set_xlabel('Čas [vzorci]')
    axs[1].set_ylabel('Amplituda')

    axs[2].plot(konv_nicle, 'r*-')
    axs[2].legend(['ničle', 'zrcaljen', 'krožni'])
    axs[2].set_title('Rob nicle')
    axs[2].set_xlabel('Čas [vzorci]')
    axs[2].set_ylabel('Amplituda')

    axs[3].plot(konv_zrcaljen, 'g*-')
    axs[3].legend([ 'zrcaljen'])
    axs[3].set_title('Rob zrcaljen')
    axs[3].set_xlabel('Čas [vzorci]')
    axs[3].set_ylabel('Amplituda')

    axs[4].plot(konv_krozni, 'b*-')
    axs[4].legend(['krožni'])
    axs[4].set_title('Rob krozni')
    axs[4].set_xlabel('Čas [vzorci]')
    axs[4].set_ylabel('Amplituda')

    plt.tight_layout()
    plt.show()