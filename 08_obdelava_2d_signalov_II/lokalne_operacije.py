
import numpy as np

def conv_2d(slika: np.ndarray, jedro: np.ndarray) -> np.ndarray:
    # dimensions
    H, W, _ = slika.shape
    N, M = jedro.shape
    polovica_N = N // 2
    polovica_M = M // 2

    konvolucija = np.zeros_like(slika)

    # iterate over each pixel
    for i in range(H):
        for j in range(W):
            for k in range(3):  # process each channel separately
                vrednost = 0.0
                # iterate over each row and column of jedro
                for p in range(N):
                    for q in range(M):
                        x = i + p - polovica_N  # row index
                        y = j + q - polovica_M  # column index
                        if x >= 0 and x < H and y >= 0 and y < W:   # check if indices are withing bounds
                            vrednost += slika[x, y, k] * jedro[p, q]
                konvolucija[i, j, k] = vrednost # assign the values

    return konvolucija


def RGB_glajenje(slika: np.ndarray, faktor: float) -> np.ndarray:
    H, W, _ = slika.shape
    polovica = int(faktor) // 2

    glajena_slika = np.zeros_like(slika)

    # iterate over each pixel
    for i in range(H):
        for j in range(W):
            for k in range(3):  # process each channel separately
                vsota = 0.0
                piksli = 0
                for p in range(i - polovica, i + polovica + 1):
                    for q in range(j - polovica, j + polovica + 1):
                        if p >= 0 and p < H and q >= 0 and q < W:
                            vsota += slika[p, q, k]
                            piksli += 1
                glajena_slika[i, j, k] = vsota / piksli     # compute the average pixel value and assign it to output image

    return glajena_slika.astype(np.uint8)


def RGB_ostrenje(slika: np.ndarray, faktor_glajenja: float, faktor_ostrenja: float) -> np.ndarray:
    glajena_slika = RGB_glajenje(slika, faktor_glajenja)
    # calculate ostrena_slika as a weighted sum of the original and 
    ostrena_slika = slika + faktor_ostrenja * (slika - glajena_slika)
    # clip pixel values to their valid range
    ostrena_slika = np.clip(ostrena_slika, 0, 255).astype(np.uint8)

    return ostrena_slika



if __name__=='__main__':
    import matplotlib.pyplot as plt
    image = plt.imread('slike/sea-breeze-apartments-pool.jpg').astype(np.uint8)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)  # example kernel
    convolved_image = conv_2d(image, kernel)

    smoothing_factor = 15.0
    smoothed_image = RGB_glajenje(image, smoothing_factor)

    smoothing_factor = 7.0
    sharpening_factor = 0.3
    sharpened_image = RGB_ostrenje(image, smoothing_factor, sharpening_factor)

    plt.figure(figsize=(10, 4))

    plt.subplot(141)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(142)
    plt.imshow(convolved_image)
    plt.title('Convolved Image')
    plt.axis('off')

    plt.subplot(143)
    plt.imshow(smoothed_image)
    plt.title('Smoothed Image')
    plt.axis('off')

    plt.subplot(144)
    plt.imshow(sharpened_image)
    plt.title('Sharpened Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()