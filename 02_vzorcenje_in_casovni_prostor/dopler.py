import numpy as np
import matplotlib.pyplot as plt

# linear interpolation between adjacant samples of vektor based on time delay and Doppler shift
def interpolation(vektor, times, vektor_fvz):
    # new time-shifted index array 
    k = np.arange(0, len(vektor), 1) - times * vektor_fvz 
    # rounding down k
    kf = np.floor(k).astype(int) 
    # (1.0-k+kf) * vektor[kf] + (k-kf) - lower adjacent point to k
    # (k-kf) * vektor[kf+1] - upper adjacent point to k
    # (kf >= 0) - values are greater or equal to 0
    return ( (1.0-k+kf) * vektor[kf] + (k-kf) * vektor[kf+1] ) * (kf >= 0)

def dopler_efekt_mono(vzorec : np.ndarray, vektor_fvz : int, oddaljenost : float, hitrost : float) -> np.ndarray:
        	            # vzorec - array containing original audio signal
                        # vektor_fvz - sampling frequency
                        # hitrost -  
    array_length = int(vektor_fvz * ((oddaljenost * 2) / hitrost))

    # 1d array with values evenly spaced with step 1
    time_vector = np.arange(array_length) / vektor_fvz

    # creates sine wave at frequency of 2000 Hz
    vektor = np.sin(2.0 * np.pi * 3000 * time_vector) 

    # position of the source
    # calculationg x coordinated of moving source at each point of time, assuming constant velocity
    x = time_vector * hitrost
    # centering source position at the origin
    x -= x.max() / 2 # subracts max(x) - 2 of each element of x
    y = np.zeros(array_length)
    z = 100.0 * np.ones(array_length)

    # 3d position of source at given time 
    position_source = np.vstack((x,y,z)).T
    # position of receiver set to 0
    position_receiver = np.zeros(3)

    # calculating Euclidean distance between source and receiver
    # used to calculate time delay, which is later used to calculated frequency shift
    dolzina = np.linalg.norm((position_source - position_receiver), axis=-1)
    
    # dolzina/343 gives time needed from source to receiver
    # interpolation function is used to shift the frequency of the input signal based on calculated time delay
    tmp = interpolation(vektor, (dolzina / 343.0), vektor_fvz)

    tmp = np.reshape(tmp, (-1,1))

    return tmp / np.max(np.abs(tmp))


if __name__ == '__main__':
    import sounddevice as sd

    #times = np.arange(int(44100.0*2.0)) / 44100.0
    #vzorec = np.sin(2.*np.pi*2000*times)

    # Define the input signal
    vzorec = np.array([0.1, 0.3, 0.2, 0.5, 0.4, 0.7, 0.8, 0.6])
    # Define the speed of the source
    hitrost = 20
    # Define the distance between the source and the observer
    oddaljenost = 700
    # Define the sampling frequency of the input signal
    vektor_fvz = 3000
    
    doppler_signal = dopler_efekt_mono(vzorec, vektor_fvz, oddaljenost, hitrost)

    fig, axs = plt.subplots(2)
    axs[0].plot(np.arange(len(vzorec)) / vektor_fvz, vzorec)
    axs[0].set(title='Original Signal')
    axs[1].plot(np.arange(len(doppler_signal)) / (vektor_fvz * (1 + hitrost/343.2)), doppler_signal)
    axs[1].set(title='Doppler-shifted Signal')
    sd.play(doppler_signal,44100)
    sd.wait()
    plt.show()