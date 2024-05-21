import numpy as np  

def analiziraj_vzorcevalno_mono(signal, dominantna_frekvenca):  # dominantna_frekvenca - frequency with the most energy

    #windowed_signal = sig * np.hamming(sig.shape[0])  # Applying Hamming window to the signal
    # calculating  discrete Fourier transform of the input signal along the first axis
    X = np.fft.fft(signal, axis=0)
    
    # find index of max value in the X array, which corresponds to the dominant frequency component of signal
    n = np.argmax(X)
    
    # calculating dominant frequency, by dividing input dominantna_frequenca 
    # by the normalized intex n divided by number of elts in the first axis of X (number of samles)
    Fvz = dominantna_frekvenca / (n / X.shape[0])
    
    return Fvz

if __name__ == '__main__':
    # time array, starts from 0, ends at 1 * 250, with step 1, divided by 250 to give values in seconds
    t = np.arange(0, 1 * 250, 1) / 250  
    
    # sinusoidal signal, with frequency of 7 Hz
    sig = 1 * np.sin(2 * np.pi * 7 * t + 0 * np.pi)
         
    sig.shape = (-1, 1)
    
    print(analiziraj_vzorcevalno_mono(sig, 7))