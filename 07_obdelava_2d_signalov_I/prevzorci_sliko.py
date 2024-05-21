import numpy as np
from scipy.ndimage import zoom, gaussian_filter
from numpy.fft import fft2, ifft2


def prevzorci_sliko(slika: np.ndarray, nova_visina: int, nova_sirina: int) -> np.ndarray:
    # apply gaussian_filter to avoid aliasing
    smoothed_image = gaussian_filter(slika, sigma=0.5)
    
    # calculate the scale factor for resizing
    scale_factor_y = nova_visina / slika.shape[0]
    scale_factor_x = nova_sirina / slika.shape[1]
    
    # check if image is greyscale or color
    if len(slika.shape) == 2:
        # greyscale image
        resized_image = zoom(smoothed_image, (scale_factor_y, scale_factor_x), order=1)
    else:
        # color image
        resized_image = zoom(smoothed_image, (scale_factor_y, scale_factor_x, 1), order=1)
    
    # return resized image
    return resized_image


def prevzorci_sliko_fft(slika: np.ndarray, nova_visina: int, nova_sirina: int) -> np.ndarray:
    # appy gaussian_filter to avoid aliasing
    smoothed_image = gaussian_filter(slika, sigma=0.5)
    
    # compute 2d fft of the smoothed image
    frequency_domain = fft2(smoothed_image)
    
    # calculate desired shape of the frequency domain
    target_shape = (nova_visina, nova_sirina) + slika.shape[2:]
    
    # adjust frequency domain to match the desired shape
    adjusted_frequency_domain = np.zeros(target_shape, dtype=np.complex128)
    # calculating indices of the center pixel in target_shape, 
    # // operator preforms integer division, ensures that center indices are rounded down to the nearest whole number
    center_y = target_shape[0] // 2
    center_x = target_shape[1] // 2
    adjusted_frequency_domain[
        :frequency_domain.shape[0], :frequency_domain.shape[1], ...
    ] = frequency_domain[
        :target_shape[0], :target_shape[1], ...
    ]
    
    # compute inverse fft to obtain resampled image
    resampled_image = np.real(ifft2(adjusted_frequency_domain))
    
    # return the resampled image 
    return resampled_image.astype(slika.dtype)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from PIL import Image

    input_image = np.array(Image.open("slike/phone_red.jpg"))
    nova_visina = 600
    nova_sirina = 400

    resampled_image = prevzorci_sliko(input_image, nova_visina, nova_sirina)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(input_image)
    axs[0].set_title("Original Image")
    axs[1].imshow(resampled_image.astype(np.uint8))
    axs[1].set_title("Resampled Image")
    plt.show()

    resampled_image_fft = prevzorci_sliko_fft(input_image, nova_visina, nova_sirina)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(input_image)
    axs[0].set_title("Original Image")
    axs[1].imshow(resampled_image_fft.astype(np.uint8))
    axs[1].set_title("Resampled Image (FFT)")
    plt.show()
