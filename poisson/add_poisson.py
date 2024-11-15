'''
Script para simular imagens tomográficas com baixa exposição

Metodologia: 

    1. Aplica a Transformada de Radon Inversa na imagem para obter as projeções (sinograma)
    2. Adiciona ruído Poisson nas projeções
    3. Aplica a Transformada de Radon (via retroprojeção filtrada) para reconstruir imagem
'''

import warnings
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.io import imshow, imsave
from skimage.util import img_as_float, img_as_ubyte, random_noise
from skimage.transform import radon, iradon, iradon_sart

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

img = imread('data/images/Phantom.png')
#img = imread('281.png')
#img = imread('ct0.png')
#img = imread('ct1.png')
#img = imread('ct2.png')
#img = imread('ct3.png')
#img = imread('ct4.tif')
#img = imread('ct5.tif')
#img = imread('ct7.tif')
#img = imread('ct8.tif')
#img = imread('ct9.tif')


nlin = img.shape[0]
ncol = img.shape[1]
n = nlin*ncol

# Normalize image to [0, 1]
y = ((img - img.min())/(img.max() - img.min()))
#y = 255*((img - img.min())/(img.max() - img.min()))

# Plot normalized image
plt.figure(1)
imshow(y, cmap='gray')
plt.axis('off')
imsave('CT.png', img_as_ubyte(y))   # pixels de 0 a 255
plt.show()

# Radon transform
theta = np.linspace(0.0, 180.0, max(img.shape), endpoint=False)
sinogram = radon(y, theta=theta)

# Plot sinogram
plt.figure(2)
plt.axis('off')
imshow(sinogram, cmap='gray')
plt.show()

# Reconstruction without noise in the projections
# filter_name: ramp, shepp-logan, cosine, hamming, hann
rec = iradon(sinogram, theta=theta)
# Normalize image
z = (255*((rec - rec.min())/(rec.max() - rec.min()))).astype(int)
error = z - img_as_ubyte(y)
print('NMSE reconstruction error:', np.sqrt(np.mean(error**2)/n))

# Plot reconstructed image (without noise)
plt.figure(3)
imshow(z, cmap='gray')
plt.axis('off')
imsave('Original.png', img_as_ubyte(z))  # pixels de 0 a 255
plt.show()

# Parâmetros para controlar a quantidade de ruído
MAX = sinogram.max()
# QUanto menor o valor de PEAK, mais ruído
#PEAK = 5000
#PEAK = 10000
PEAK = 15000
sinogram_n = np.random.poisson(sinogram/MAX*PEAK)/PEAK*MAX 

# Plot sinogram
plt.figure(4)
plt.axis('off')
imshow(sinogram_n, cmap='gray')
plt.show()

# Reconstruction with noise in the projections
# filters = ramp, shepp-logan, cosine, hamming, hann
rec_n = iradon(sinogram_n, theta=theta)
# Normalize noisy image
x = (255*((rec_n - rec_n.min())/(rec_n.max() - rec_n.min()))).astype(int)
error = x - img_as_ubyte(y)
print('NMSE reconstruction error (with Poisson noise):', np.sqrt(np.mean(error**2)/n))

# Plot reconstructed image (with noise)
plt.figure(5)
imshow(x, cmap='gray')
plt.axis('off')
imsave('Noisy.png', img_as_ubyte(x))  # pixels de 0 a 255
plt.show()
