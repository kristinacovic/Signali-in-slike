import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, binary_dilation

def open(slika: np.ndarray, jedro: np.ndarray) -> np.ndarray:
    eroded_image = binary_erosion(slika, structure=jedro)   # erodes away white regions
    opened_image = binary_dilation(eroded_image, structure=jedro)   # dilates (expands) remaining region
    return opened_image

def close(slika: np.ndarray, jedro: np.ndarray) -> np.ndarray:
    dilated_image = binary_dilation(slika, structure=jedro)      # expands white regions
    closed_image = binary_erosion(dilated_image, structure=jedro)   # erodes away small foregrounds that have been created with dilation
    return closed_image

def hit_miss(slika: np.ndarray, jedro_hit: np.ndarray, jedro_miss: np.ndarray) -> np.ndarray:
    eroded_hit = binary_erosion(slika, structure=jedro_hit)
    eroded_miss = binary_erosion(~slika, structure=jedro_miss)
    # taking intersection of hit and miss
    hit_miss_image = eroded_hit & eroded_miss
    return hit_miss_image

if __name__ == '__main__':
    # grayscale to binary
    image = plt.imread('slike/Bikesgray.jpg')
    threshold = 100  

    binary_image = image > threshold

    kernel_size_open = 5
    kernel_size_close = 5
    kernel_size_hit = 5
    kernel_size_miss = 5

    jedro_open = np.ones((kernel_size_open, kernel_size_open), bool)
    jedro_close = np.ones((kernel_size_close, kernel_size_close), bool)
    jedro_hit = np.random.choice([True, False], size=(kernel_size_hit, kernel_size_hit))
    jedro_miss = np.random.choice([True, False], size=(kernel_size_miss, kernel_size_miss))


    opened_image = open(binary_image, jedro_open)
    closed_image = close(binary_image, jedro_close)
    hit_miss_image = hit_miss(binary_image, jedro_hit, jedro_miss)

    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')

    plt.subplot(2, 2, 2)
    plt.imshow(opened_image)
    plt.title('Opened Image')

    plt.subplot(2, 2, 3)
    plt.imshow(closed_image)
    plt.title('Closed Image')

    plt.subplot(2, 2, 4)
    plt.imshow(hit_miss_image)
    plt.title('Hit-Miss Image')

    plt.tight_layout()
    plt.show()