'''
Script para construir o phantom de Shepp Logan a partir de projeções ruidosas
'''

import warnings
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import skimage.io as skio
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, rescale
from skimage.util import random_noise

# Para evitar mensagens de warning desnecessárias
warnings.simplefilter(action='ignore')

# Carrega phantom
image = shepp_logan_phantom()

# Plota imagem
plt.figure(1)
plt.imshow(image, cmap=plt.cm.Greys_r)
plt.axis('off')
plt.show()
skio.imsave('Phantom.png', image, plugin='pil')

# Aplica transformada de Radon (obter as projeções - sinograma)
theta = np.linspace(0., 180., max(image.shape), endpoint=False)
sinogram = radon(image, theta=theta)

# plota sinograma
plt.figure(2)
plt.imshow(sinogram, cmap=plt.cm.Greys_r)
plt.axis('off')
plt.show()
skio.imsave('Sinogram.png', sinogram.astype(int), plugin='pil')

#sino_n = random_noise(sinogram, mode='poisson', clip=False)
#PEAK = sinogram.max()
#PEAK = 2550
#PEAK = 3000		# Quanto maior, menos ruído Poisson
PEAK = 5000
sino_n = np.random.poisson(sinogram/255.0*PEAK)/PEAK*255 

# plota sinograma ruidoso
plt.figure(3)
plt.imshow(sino_n, cmap=plt.cm.Greys_r)
plt.axis('off')
plt.show()
skio.imsave('Noisy_Sinogram.png', sino_n.astype(int), plugin='pil')

# Imagem reconstruída
reconstruction_fbp = iradon(sino_n, theta=theta, filter_name='ramp')

# plota sinograma ruidoso
plt.figure(4)
plt.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
plt.axis('off')
plt.show()
output = (reconstruction_fbp - reconstruction_fbp.min())/(reconstruction_fbp.max() - reconstruction_fbp.min())
skio.imsave('Noisy_Phantom.png', output, plugin='pil')
