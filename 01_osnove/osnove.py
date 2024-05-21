import numpy as np
import scipy.io.wavfile as wavfile
import timeit

# returns a vector of random numbers evenly distrubuted betweeen 0 and 1
def generiraj_vektor_nakljucno(stevilo_vzorcev: int) -> np.ndarray:
    return np.random.uniform(size=(stevilo_vzorcev, 1))
  
def beri_zvok_mono (pot_do_datoteke: str) -> np.ndarray:
    Fs, y = wavfile.read(pot_do_datoteke)

    # if audio data is stereo format
    if (y.ndim > 1):
        vektor = np.mean(y, axis=1)
        vektor = vektor.reshape((-1, 1))
        return vektor / np.abs(vektor).max()
    
    vektor = y.reshape((-1,1))
    return vektor / np.abs(vektor).max()

def beri_zvok_stereo(pot_do_datoteke: str) -> np.ndarray:
    Fs, y = wavfile.read(pot_do_datoteke)

    # for audio data is mono format
    if(y.ndim == 1):
        # creates 2D array
        vektor = np.zeros((y.size,2))
        for n in range(y.size):
            vektor[n][0] = y[n] # setting values of first column
            vektor[n][1] = y[n] # setting values of second column

        return vektor / np.abs(vektor).max()
    
    else:
        vektor = y
        vektor = np.reshape(vektor,(-1,2))
        return vektor / np.abs(vektor).max()
    

def normaliziraj(vektor: np.ndarray) -> np.ndarray:
    vektor = vektor.astype(np.float64)
    # storing shape of vector in oblika
    oblika = vektor.shape
    # reshaped to 1d
    tmp = np.reshape(vektor, (-1))
    tmp = tmp.astype(np.float64)
    # max_value initialized to 0
    max_value = 0
    # iterates over each elt of array and sets max. to abs. value of all elts
    for x in tmp:
        if abs(x) > max_value:
            max_value = abs(x)

    # divides each elt of array with max_value
    i = 0
    while i < tmp.size:
        tmp[i] = tmp[i] / max_value
        i += 1

    # reshaping back to original shape
    vektor = np.reshape(tmp, oblika)
    # setting outcome to be of type np.float64
    vektor = vektor.astype(np.float64)
            
    return vektor


def normaliziraj_vektorsko(vektor: np.ndarray) -> np.ndarray:
    vektor = vektor / np.abs(vektor).max()
    vektor = vektor.astype(np.float64)
    
    return vektor


def testiraj_cas_izvajanja(vektor: np.ndarray) -> np.ndarray:
    start = timeit.default_timer()
    normaliziraj(vektor)
    time_normaliziraj = timeit.default_timer() - start

    start2 = timeit.default_timer()
    normaliziraj_vektorsko(vektor)
    time_normaliziraj_vektorsko = timeit.default_timer() - start2

    return np.array([[time_normaliziraj, time_normaliziraj_vektorsko]])

if __name__ == '__main__':
    print(testiraj_cas_izvajanja(generiraj_vektor_nakljucno(1000000)))
