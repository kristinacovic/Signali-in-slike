import numpy as np

def RGB_v_HSV(slika: np.ndarray) -> np.ndarray:
    # dimensions of the image
    H, W, _ = slika.shape
     # array of zeros with same shape as slika
    hsv_slika = np.zeros_like(slika, dtype=np.uint8) 

    #iterate over each pixel of image
    for i in range(H):
        for j in range(W):
            # getting rgb values of current pixel
            r, g, b = slika[i, j]
            # normalize rgb values to the range[0,1]
            r, g, b = r / 255.0, g / 255.0, b / 255.0

            # find max and min values among rgb components
            cmax = max(r, g, b)
            cmin = min(r, g, b)

            # difference betwen max and min
            delta = cmax - cmin 
            
            # calculating h value (hue - odtenek)
            if delta == 0:
                h = 0
            elif cmax == r:
                h = 60 * (((g - b) / delta) % 6)
            elif cmax == g:
                h = 60 * (((b - r) / delta) + 2)
            else:
                h = 60 * (((r - g) / delta) + 4)
            
            # calculating v value
            v = cmax * 255
            # calculating s (saturation) value
            s = 0 if cmax == 0 else (delta / cmax) * 255
            
            # assign the calculated HSV values to the corresponding pixel in the output image
            hsv_slika[i, j] = np.uint8([h / 2, s, v])
    
    # return resulting hsv image
    return hsv_slika

def HSV_v_RGB(slika: np.ndarray) -> np.ndarray:
    # get dimensions of the image
    H, W, _ = slika.shape
    # zeros array, with size and shape of slika
    rgb_slika = np.zeros_like(slika, dtype=np.uint8)
    
    # iterate over the each pixel of image
    for i in range(H):
        for j in range(W):
            # get hsv values for the current pixel
            h, s, v = slika[i, j]
            # adjust the hue value
            h = h * 2
            # normalize saturation and value to the range [1,0]
            s = s / 255.0
            v = v / 255.0
            
            # calculate chroma, intermediate values and m value
            c = v * s
            x = c * (1 - abs((h / 60) % 2 - 1))
            m = v - c
            
            # determine rgb values based on the hue range
            if 0 <= h < 60:
                r, g, b = c, x, 0
            elif 60 <= h < 120:
                r, g, b = x, c, 0
            elif 120 <= h < 180:
                r, g, b = 0, c, x
            elif 180 <= h < 240:
                r, g, b = 0, x, c
            elif 240 <= h < 300:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            
            # assign the calculated rgb values to the corresponding pixel in the output
            rgb_slika[i, j] = np.uint8([(r + m) * 255, (g + m) * 255, (b + m) * 255])
    
    return rgb_slika

def RGB_v_YCbCr(slika: np.ndarray) -> np.ndarray:
    # get height and width of slika
    H, W, _ = slika.shape
    # create output array for YCbCr converted image
    ycbcr_slika = np.zeros_like(slika, dtype=np.uint8)
    
    #iterate over each pixel
    for i in range(H):
        for j in range(W):
            r, g, b = slika[i, j]   # get the rgb values of the current pixel
            # Apply YCbCr conversion formulas to calculate Y, Cb, and Cr values
            y = 0.299 * r + 0.587 * g + 0.114 * b
            cb = -0.169 * r - 0.331 * g + 0.5 * b + 128
            cr = 0.5 * r - 0.419 * g - 0.081 * b + 128
            
            ycbcr_slika[i, j] = np.uint8([y, cb, cr])   # assign the calculated ycbcr values to the output
    
    return ycbcr_slika

def YCbCr_v_RGB(slika: np.ndarray) -> np.ndarray:
    H, W, _ = slika.shape   # get the height and width
    rgb_slika = np.zeros_like(slika, dtype=np.uint8)
    
    for i in range(H):  # iterate over each row
        for j in range(W):  # over each column
            # apply formulas to get rgb values
            y, cb, cr = slika[i, j]
            r = y + 1.4 * (cr - 128)
            g = y - 0.343 * (cb - 128) - 0.711 * (cr - 128)
            b = y + 1.765 * (cb - 128)
            
            rgb_slika[i, j] = np.uint8([r, g, b])
    
    return rgb_slika


if __name__ == '__main__':
    import matplotlib.pyplot as pyplot

    slika = pyplot.imread('slike/00070.ppm')

    #   RGB to HSV
    slikaHSV = RGB_v_HSV(slika)

    pyplot.figure()
    pyplot.imshow(slikaHSV)
    pyplot.title('HSV Image')
    pyplot.axis('off')

    # HSV to RGB conversion
    slikaRGB = HSV_v_RGB(slikaHSV)
    pyplot.figure()
    pyplot.imshow(slikaRGB)
    pyplot.title('RGB Image')
    pyplot.axis('off')

    pyplot.show()




    
    # RGB to YCbCr conversion
    slikaYCbCr = RGB_v_YCbCr(slika)
    slikaYCbCr = np.clip(slikaYCbCr, 0, 255)


    pyplot.figure()
    pyplot.imshow(slikaYCbCr)
    pyplot.title('YCbCr Image')
    pyplot.axis('off')

    # YCbCr to RGB conversion
    slikaRGB = YCbCr_v_RGB(slikaYCbCr)

    # Clip the RGB image values to the range [0, 255]
    slikaRGB = np.clip(slikaRGB, 0, 255)

    pyplot.figure()
    pyplot.imshow(slikaRGB)
    pyplot.title('RGB Image')
    pyplot.axis('off')

    pyplot.show()
