import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import date
import bm3d
from skimage import exposure
from skimage.io import imread
from skimage.transform import radon, iradon
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2gray
from skimage.io import imshow, imsave
from skimage.util import img_as_float, img_as_ubyte, random_noise

import os
import warnings
import numpy as np

warnings.simplefilter(action='ignore')


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


input_path = "data/images/"
output_path = "data/results"
execution_date = date.today()

patch_kw = dict(patch_size=5, patch_distance=6, channel_axis=None) # 5x5 patches  # 13x13 search area
noise_levels = [13000, 7500, 3000] # Níveis de ruído Poisson (alto a baixo)
apply_anscombe_flag = [True, False] # Controle para aplicar a transformada Anscombe
# sigma_multipliers = np.arange(0.8, 2.5, 0.1) # Multiplicadores do desvio padrão do ruído
ta_sigma_multipliers = np.arange(0.01, 0.25, 0.005) # Multiplicadores do desvio padrão do ruído
not_ta_sigma_multipliers = np.arange(0.1, 3.0, 0.1) # Multiplicadores do desvio padrão do ruído
images = os.listdir(input_path)
images = ["ct3.png", "ct4.png", "ct6.png"]




for image_name in images:
    
    # Salva os resultados para cada imagem
    results = []
    
    # Carrega e normaliza a imae=gem de entrada
    img = imread(input_path + image_name)
    img = img.astype(np.uint8)
    img_normalized = (1*((img - img.min())/(img.max() - img.min()))) # píxeis de 0 a 1
    img = (255*((img - img.min())/(img.max() - img.min()))).astype(int)
    
    # Aplica a Transformada de Radon (Sinograma)
    theta = np.linspace(0.0, 180.0, max(img.shape), endpoint=False)
    sinogram = radon(img_normalized, theta=theta)
    
    # Executa apra diferentes níveis de ruído
    for noise_level in noise_levels:

        # Adiciona ruído poisson no sinograma da imagem
        sinogram_noisy = np.random.poisson(sinogram / sinogram.max() * noise_level) / noise_level * sinogram.max()

        # Reconstrução da imagem usando a Transformada Inversa de Radon
        rec_noisy = iradon(sinogram_noisy, theta=theta)
        rec_noisy = (255*((rec_noisy - rec_noisy.min())/(rec_noisy.max() - rec_noisy.min()))).astype(int)
        rec_noisy[np.where(img_normalized==0)] = 0
        
        imsave(f'{output_path}/{Path(image_name).stem}/{noise_level}/{Path(image_name).stem}_noisy.png',  img_as_ubyte(rec_noisy)) # píxeis de 0 a 255
        
        # Estima o desvio padrão da imagem ruidosa
        sigma_est = np.mean(estimate_sigma(rec_noisy, channel_axis=None))
                
        # Cálcula as métricas em relação à imagem ruidosa
        psnr_noisy = peak_signal_noise_ratio(img.astype(np.uint8), rec_noisy.astype(np.uint8))
        ssim_noisy, ssim_map = structural_similarity(img.astype(np.uint8), rec_noisy.astype(np.uint8), full=True)


        for apply_transform in apply_anscombe_flag:
            
            # Aplica a transforma de anscombe se apply_transform is True
            rec_noisy_transformed = generalized_anscombe_transform(rec_noisy.astype(float), sigma_est) if apply_transform else rec_noisy.astype(float).copy()
            
            sigma_multipliers = ta_sigma_multipliers if apply_transform else not_ta_sigma_multipliers
            
            for sigma_multiplier in sigma_multipliers:
            
                # Filtro NL-Means
                img_denoised_nlm = denoise_nl_means(rec_noisy_transformed, h=sigma_multiplier * sigma_est, fast_mode=True, **patch_kw)
                img_denoised_nlm = exact_inverse_anscombe_transform(img_denoised_nlm, sigma_est) if apply_transform else img_denoised_nlm
                
                # Calcula PSNR e SSIM da imagem filtrada com NL-Means
                psnr_nlm = peak_signal_noise_ratio(img.astype(np.uint8), img_denoised_nlm.astype(np.uint8))
                ssim_nlm, _ = structural_similarity(img.astype(np.uint8), img_denoised_nlm.astype(np.uint8), full=True)
                
                # Salva os resultados NL-Means
                results.append((image_name, noise_level, sigma_est, sigma_multiplier, apply_transform, 'nlmeans', psnr_noisy, ssim_noisy, psnr_nlm, ssim_nlm))
                
                # Filtro BM3D
                img_denoised_bm3d = bm3d.bm3d(rec_noisy_transformed, sigma_psd=sigma_multiplier * sigma_est, stage_arg=bm3d.BM3DStages.ALL_STAGES)
                img_denoised_bm3d = exact_inverse_anscombe_transform(img_denoised_bm3d, sigma_est) if apply_transform else img_denoised_bm3d
                
                # Calcula PSNR e SSIM da imagem filtrada com BM3D
                psnr_bm3d = peak_signal_noise_ratio(img.astype(np.uint8), img_denoised_bm3d.astype(np.uint8))
                ssim_bm3d, _ = structural_similarity(img.astype(np.uint8), img_denoised_bm3d.astype(np.uint8), full=True)
                
                # Salva os resultados BM3D
                results.append((image_name, noise_level, sigma_est, sigma_multiplier, apply_transform, 'bm3d', psnr_noisy, ssim_noisy, psnr_bm3d, ssim_bm3d))
                
                # Salva as imagens
                if apply_transform:
                    nlm_path = f'{output_path}/{Path(image_name).stem}/{noise_level}/{Path(image_name).stem}_nlm_ta_{sigma_multiplier:.2f}.png'
                    bm3d_path = f'{output_path}/{Path(image_name).stem}/{noise_level}/{Path(image_name).stem}_bm3d_ta_{sigma_multiplier:.2f}.png'
                else:
                    nlm_path = f'{output_path}/{Path(image_name).stem}/{noise_level}/{Path(image_name).stem}_nlm_{sigma_multiplier:.2f}.png'
                    bm3d_path = f'{output_path}/{Path(image_name).stem}/{noise_level}/{Path(image_name).stem}_bm3d_{sigma_multiplier:.2f}.png'
                imsave(nlm_path, img_denoised_nlm.astype(np.uint8))
                imsave(bm3d_path, img_denoised_bm3d.astype(np.uint8))

    csv_path = f'{output_path}/{Path(image_name).stem}/{execution_date}.csv'
    
    with open(csv_path, mode='w') as file:

        file.write("image_name,peak,sigma_est,sigma_multiplier,anscombe_transform,filter,psnr_noisy,ssim_noisy,psnr_denoised,ssim_denoised\n")
        
        for item in results:
            line = ",".join(map(str, item))  
            file.write(line + "\n")