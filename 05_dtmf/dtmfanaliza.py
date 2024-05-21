import numpy as np
from scipy.io import wavfile

#DTMF table for recognizing characters
DTMF = {(697, 1209): "1", 
        (697, 1336): "2", 
        (697, 1477): "3", 
        (770, 1209): "4", 
        (770, 1336): "5", 
        (770, 1477): "6", 
        (852, 1209): "7", 
        (852, 1336): "8", 
        (852, 1477): "9", 
        (941, 1209): "*", 
        (941, 1336): "0", 
        (941, 1477): "#"}

def analiziraj_dtmf(signal: np.ndarray, vzorcevalna_frekvenca: int, min_cas_ton: float, min_cas_pavza: float) -> np.ndarray:
    
    # empty string for storing the detected characters     
    vec = ""

    # calculating minimum number of samples required for a tone and a pause    
    min_tone_samples = int(vzorcevalna_frekvenca * min_cas_ton / 1000)
    min_pause_samples = int(vzorcevalna_frekvenca * min_cas_pavza / 1000)

    # empty string for storing the current detected character    
    current_char = ""
    
    # loop through the signal in steps of min_tone_samples samples
    for i in range(0, len(signal), min_tone_samples):

        # compute the FFT of the current segment of the signal (i to i + min_tone_samples)
        sigFFT = np.fft.fft(signal[i:i+min_tone_samples])

        # compute the corresponding frequency values for the FFT coefficients
        freqs = np.fft.fftfreq(signal[i:i+min_tone_samples].size, 1 / vzorcevalna_frekvenca)

        # finding the range of low frequencies

        lower_freq_min = np.where(freqs > 600)[0]
        if lower_freq_min.size > 0:
            lower_freq_min = lower_freq_min[0].item()
        else:
            lower_freq_min = 0

        if (freqs > 1000).any():
            lower_freq_max = np.where(freqs > 1000)[0][0].item()
        else:
            lower_freq_max = len(freqs) - 1

        lower_freqs = freqs[int(lower_freq_min):int(lower_freq_max)]

        lower_amps = abs(sigFFT.real[lower_freq_min:lower_freq_max])

        # finding the dominant low frequency
        lower_freq = None
        if len(lower_amps) > 0:
            lower_freq = lower_freqs[np.where(lower_amps == max(lower_amps))[0][0]]

        if lower_freq is None:
            continue

        # initialize variables for finding the closest DTMF low frequency
        offset = 10
        closest_low_freq = 0

        # loop through the DTMF low frequencies and find the closest match to the current low frequency
        for f in [697, 770, 852, 941] :
            if abs(lower_freq - f) < offset :
                offset = abs(lower_freq - f)
                closest_low_freq = f
        # store the closest DTMF low frequency
        lower_freq = closest_low_freq

        # finding the range of high frequencies
        upper_freq_min = np.where(freqs > 1100)[0]
        if upper_freq_min.size > 0:
            upper_freq_min = upper_freq_min[0]
        else:
            return None
        if (freqs > 1700).any():
            upper_freq_max = np.where(freqs > 1700)[0][0]
        else:
            upper_freq_max = len(freqs) - 1

        upper_freqs = freqs[upper_freq_min:upper_freq_max]
        # extract the high frequency range and corresponding amplitudes
        upper_amps = abs(sigFFT.real[upper_freq_min:upper_freq_max])
        # finding the dominant high frequency
        upper_freq = None
        if len(upper_amps) > 0:
                upper_freq = upper_freqs[np.where(upper_amps == max(upper_amps))[0][0]]

        # initialize variables for finding the closest DTMF high frequency
        offset = 10
        closest_high_freq = 0

        # loop through the DTMF high frequencies and find the closest match to the current high frequency
        for f in [1209, 1336, 1477] :
            if abs(upper_freq - f) < offset :
                offset = abs(upper_freq - f)
                closest_high_freq = f
        # store the closest DTMF high frequency
        upper_freq = closest_high_freq

        # checking if the current tone has ended and a pause has started
        if len(current_char) > 0 and (i - last_tone_end) >= min_pause_samples:
            vec += DTMF[(current_freq, current_high_freq)]
            current_char = ""

        # checking if a new tone has started
        if upper_freq and lower_freq and not current_char:
            current_freq = lower_freq
            current_high_freq = upper_freq
            current_char = DTMF[(current_freq, current_high_freq)]
            last_tone_end = i

    # appending the last detected character if the signal ends on a tone
    if len(current_char) > 0:
        vec += current_char

    return np.array([char for char in vec])




if __name__ == '__main__':
    print("Modul za DTMF analizo!")
    
    #Fvz, sig = wavfile.read('dtmf_123456789_0__min_pulse_0.1_min_pause_0.1.wav')
    #Fvz, sig = wavfile.read('dtmf_124679_min_pulse_0.1_min_pause_0.1.wav')
    #Fvz, sig = wavfile.read('dtmf_123_min_pulse_0.2_min_pause_0.1_noise_low.wav')
    #Fvz, sig = wavfile.read('dtmf_123_min_pulse_0.2_min_pause_0.1_noise_med.wav')
    #Fvz, sig = wavfile.read('dtmf_123_min_pulse_0.2_min_pause_0.1_noise_high.wav')
    #vec = analiziraj_dtmf(sig,Fvz, 100, 100)

    #print(vec)Ž

    vzorcevalna_frekvenca, signal = wavfile.read('dtmf_123_min_pulse_0.2_min_pause_0.1_noise_high.wav')

    #analyze the signal and print the detected characters
    min_cas_ton = 200 # in milliseconds
    min_cas_pavza = 100 # in milliseconds
    detected_chars = analiziraj_dtmf(signal, vzorcevalna_frekvenca, min_cas_ton, min_cas_pavza)
    print(detected_chars)

