import warnings
import numpy as np
import matplotlib.pyplot as plt
import bm3d
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import data
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.io import imshow, imsave
from skimage.util import img_as_float, img_as_ubyte, random_noise
from skimage.transform import radon, iradon, iradon_sart
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

def anscombe_transform(image):
    """Apply the Anscombe transform to the image."""
    return 2 * np.sqrt(image + 3.0/8.0)

def inverse_anscombe_transform(transformed_image):
    """Apply the inverse Anscombe transform to the image."""
    return (transformed_image / 2) ** 2 - 3.0/8.0

# Para ruído Poisson-Gaussiano essa é a verdadeira Transformada de Anscombe (generalizada)
# REF: https://github.com/broxtronix/pymultiscale/blob/master/pymultiscale/anscombe.py
def generalized_anscombe_transform(image, sigma=0, mu=0, gain=1.0): # sigma é o desvio padrão do ruído
    """Apply the Anscombe transform to the image."""
    y = gain*image + (gain**2)*3.0/8.0 + sigma**2 - gain*mu
    return (2.0/gain)*np.sqrt(np.maximum(y, 0.0))

# Para ruído Poisson-Gaussiano essa é a verdadeira Transformada Inversa de Anscombe (generalizada)
# REF: https://github.com/broxtronix/pymultiscale/blob/master/pymultiscale/anscombe.py
# REF: https://en.wikipedia.org/wiki/Anscombe_transform
def exact_inverse_anscombe_transform(transformed_image, sigma=0, mu=0, gain=1.0): # sigma é o desvio padrão do ruído
    """Apply the inverse Anscombe transform to the image."""
    exact_inverse = np.power(transformed_image/2.0, 2.0) + 1.0/4.0*np.sqrt(3.0/2.0)*np.power(transformed_image, -1.0) - 11.0/8.0*np.power(transformed_image, -2.0) + 5.0/8.0*np.sqrt(3.0/2.0)*np.power(transformed_image, -3.0) - 1.0/8.0 - sigma**2
    exact_inverse = np.maximum(0.0, exact_inverse)
    return exact_inverse

########################################
# Início 
########################################
img = imread('data/images/ct3.png')
#img = imread('ct4.png')
#img = imread('ct6.png')
#img = imread('Cancerous184.jpg')
#img = imread('Cancerous177.jpg')


# Checa se imagem é monocromática (jpg em geral pode vir com mais de um canal)
if len(img.shape) > 2:
    img = rgb2gray(img)   # valores convertidos ficam entre 0 e 1
    img = 255*img

img = img.astype(np.uint8)              # Converte para uint8    

nlin = img.shape[0]
ncol = img.shape[1]
n = nlin*ncol

# Normalize image
y = (255*((img - img.min())/(img.max() - img.min()))).astype(int)
y = ((img - img.min())/(img.max() - img.min()))
imsave('CT.png', img_as_ubyte(y))   # pixels de 0 a 255

# Radon transform
theta = np.linspace(0.0, 180.0, max(img.shape), endpoint=False)
sinogram = radon(y, theta=theta)

# Reconstruction without noise in the projections
# filters = ramp, shepp-logan, cosine, hamming, hann
rec = iradon(sinogram, theta=theta)
#rec = iradon(sinogram, theta=theta, filter_name='shepp-logan')
z = (255*((rec - rec.min())/(rec.max() - rec.min()))).astype(int)
# Optional: Make the background zero (black)
z[np.where(y==0)] = 0
imsave('REC.png', img_as_ubyte(z))  # pixels de 0 a 255

# Adiciona ruído Poisson
# Quanto maior, menos ruído Poisson
MAX = sinogram.max()
#PEAK = 5000
#PEAK = 10000
PEAK = 13000
sinogram_n = np.random.poisson(sinogram/MAX*PEAK)/PEAK*MAX 

# Reconstruction with noise in the projections
# filters = ramp, shepp-logan, cosine, hamming, hann
rec_n = iradon(sinogram_n, theta=theta)
#rec_n = iradon(sinogram_n, theta=theta, filter_name='shepp-logan')
x = (255*((rec_n - rec_n.min())/(rec_n.max() - rec_n.min()))).astype(int)
# Optional: Make the background zero (black)
x[np.where(y==0)] = 0
imsave('REC_n.png', img_as_ubyte(x))  # pixels de 0 a 255


################################################################
# Nesse ponto x é a imagem sem ruído e z é a imagem com ruído
################################################################
psnr_noisy = peak_signal_noise_ratio(z.astype(np.uint8), x.astype(np.uint8), )
print('PSNR imagem ruidosa: ', psnr_noisy)

# Inicializa parâmetros
sigma_multipliers = np.arange(0.1, 2.5, 0.1) 
# Estima o desvio padrão da imagem ruidosa
sigma_est = estimate_sigma(x.astype(float), channel_axis=None)

# NLM padrão
for sigma_multiplier in sigma_multipliers:
    img_denoised_nlm = denoise_nl_means(x.astype(float), h=sigma_multiplier*sigma_est, fast_mode=True, patch_size=4, patch_distance=7, channel_axis=None)
    psnr_nlm = peak_signal_noise_ratio(z.astype(np.uint8), img_denoised_nlm.astype(np.uint8))
    print('PSNR imagem NLM: ', psnr_nlm)
    #ssim_nlm = structural_similarity(z.astype(np.uint8), img_denoised_nlm.astype(np.uint8))
    #print('SSIM imagem NLM: ', ssim_nlm)
    imsave('NLM_'+str(sigma_multiplier)+'.png', img_denoised_nlm.astype(np.uint8))

print()

# TA + NLM 
# Inicializa parâmetros
sigma_multipliers = np.arange(0.01, 0.2, 0.005) 
TA = generalized_anscombe_transform(x.astype(float), sigma_est)
for sigma_multiplier in sigma_multipliers:
    img_denoised_nlm_ta = denoise_nl_means(TA, h=sigma_multiplier*sigma_est, fast_mode=True, patch_size=4, patch_distance=7, channel_axis=None)
    ITA = exact_inverse_anscombe_transform(img_denoised_nlm_ta, sigma_est)
    psnr_nlm_ta = peak_signal_noise_ratio(z.astype(np.uint8), ITA.astype(np.uint8))
    print('PSNR imagem NLM + TA: ', psnr_nlm_ta)
    #ssim_nlm_ta = structural_similarity(z.astype(np.uint8), ITA.astype(np.uint8))
    #print('SSIM imagem NLM + TA: ', ssim_nlm_ta)
    imsave('NLM_TA_'+str(sigma_multiplier)+'.png', img_denoised_nlm.astype(np.uint8))