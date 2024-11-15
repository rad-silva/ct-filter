import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv('data/results/2024-10-18.csv')

df['h'] = df['sigma_est'] * df['sigma_multiplier']

nlmeans = df[df['filter'] == 'nlmeans']
bm3d = df[df['filter'] == 'bm3d']

nlmeans_anscombe = nlmeans[nlmeans['anscombe_transform'] == True]
nlmeans_no_anscombe = nlmeans[nlmeans['anscombe_transform'] == False]
bm3d_anscombe = bm3d[bm3d['anscombe_transform'] == True]
bm3d_no_anscombe = bm3d[bm3d['anscombe_transform'] == False]

# Função para gerar e salvar os gráficos
def plot_and_save_metrics(peak, image_name, filter_name, df_no_anscombe, df_anscombe, output_dir):
    plt.figure(figsize=(10, 6))

    # Gráficos de PSNR
    plt.subplot(2, 1, 1)
    plt.plot(df_no_anscombe['h'], df_no_anscombe['psnr_denoised'], label=f'{filter_name} sem Anscombe', marker='o')
    plt.plot(df_anscombe['h'], df_anscombe['psnr_denoised'], label=f'{filter_name} com Anscombe', marker='o')
    plt.xlabel('h')
    plt.ylabel('PSNR')
    plt.title(f'PSNR vs h ({filter_name}) - Peak: {peak}, Image: {image_name}')
    plt.legend()
    plt.grid(True)

    # Gráficos de SSIM
    plt.subplot(2, 1, 2)
    plt.plot(df_no_anscombe['h'], df_no_anscombe['ssim_denoised'], label=f'{filter_name} sem Anscombe', marker='o')
    plt.plot(df_anscombe['h'], df_anscombe['ssim_denoised'], label=f'{filter_name} com Anscombe', marker='o')
    plt.xlabel('h')
    plt.ylabel('SSIM')
    plt.title(f'SSIM vs h ({filter_name}) - Peak: {peak}, Image: {image_name}')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Criar o diretório de saída se não existir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Salvar o gráfico
    output_path = os.path.join(output_dir, f'{image_name}_{peak}_{filter_name}.png')
    plt.savefig(output_path)
    plt.close()

# Diretório para salvar os gráficos
output_dir = 'data/graphics/'

# Iterar por 'image_name' e 'peak'
for image_name in df['image_name'].unique():
    df_image = df[df['image_name'] == image_name]
    
    for peak_value in df_image['peak'].unique():
        nlmeans_peak_no_anscombe = nlmeans_no_anscombe[(nlmeans_no_anscombe['peak'] == peak_value) & (nlmeans_no_anscombe['image_name'] == image_name)]
        nlmeans_peak_anscombe = nlmeans_anscombe[(nlmeans_anscombe['peak'] == peak_value) & (nlmeans_anscombe['image_name'] == image_name)]
        bm3d_peak_no_anscombe = bm3d_no_anscombe[(bm3d_no_anscombe['peak'] == peak_value) & (bm3d_no_anscombe['image_name'] == image_name)]
        bm3d_peak_anscombe = bm3d_anscombe[(bm3d_anscombe['peak'] == peak_value) & (bm3d_anscombe['image_name'] == image_name)]

        # Gerar e salvar os gráficos para NL-Means e BM3D
        plot_and_save_metrics(peak_value, image_name, 'NL-Means', nlmeans_peak_no_anscombe, nlmeans_peak_anscombe, output_dir)
        plot_and_save_metrics(peak_value, image_name, 'BM3D', bm3d_peak_no_anscombe, bm3d_peak_anscombe, output_dir)
