import pywt
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
from skimage import data
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.io import imread

# load the image
url = 'https://cdn.myportfolio.com/95f166b3998854f42a9e60b34a4881cf/2fe95e59-11f1-451e-bb35-1477f635cc94_rw_3840.jpg?h=5b8f52008965eb0f9600532f04333e28'
response = requests.get(url)
image = Image.open(BytesIO(response.content))

# rescale the image
#image = rescale(image, 0.5, anti_aliasing=None , multichannel=True, shape=)

# convert to grayscale
image = np.mean(image, axis=2)

# apply wavelet transform
coeffs = pywt.wavedec2(image, 'db1', level=3)

# apply fourier transform
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

# compare results
plt.subplot(121),plt.imshow(image, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()