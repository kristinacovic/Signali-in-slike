import numpy as np
import matplotlib.pyplot as plt

def generiraj_ton_mono(cas : float, vzorcevalna_frekvenca : int, bitna_locljivost: int, frekvenca_tona : float) -> np.ndarray:
    
    max_value = (pow(2,bitna_locljivost)/2 - 1)
    # min_value = -(pow(2,bitna_locljivost)/2)

    if max_value < 128:
        data_type = "int8"
    elif max_value < 32768:
        data_type = "int16"
    elif max_value < 2147483648:
        data_type = "int32"
    # ostalo je int64
    else:
        data_type = "int64"

    # for time axis 
    time_array = np.arange(0, cas * vzorcevalna_frekvenca, 1, dtype = data_type) / vzorcevalna_frekvenca

    # creates sine wave with specified frequency and max amplitude
    # 2*np.pi converts frequency from Hz to radians per second
    sine_wave = np.sin(2 * np.pi * frekvenca_tona * time_array)

    # scaling the result, generated result will be between -max_value and max_value
    sine_wave = max_value * sine_wave

    vektor = np.zeros((len(sine_wave),1), dtype=data_type)
    i=0
    while i < len(sine_wave):       # begins from i = 0, and repeats until i is equal to length of sine_wave
         # value of sine_wave is assigned to corresponding element of vektor[i]
        vektor[i] = sine_wave[i]   
        i += 1    #increases value of i, moves to the next element 

    return vektor

if __name__ == '__main__':
                            # cas, Fvz, bitna_locljivost, freekvenca tona
    #tone = generiraj_ton_mono(1, 10, 8, 5)
    tone = generiraj_ton_mono(3, 441, 32, 440)

    plt.xlabel("cas v sekundah")
    plt.ylabel("vrednost")

    plt.plot(tone)

    plt.show()