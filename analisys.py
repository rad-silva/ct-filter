import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/solutions/2024-10-03.csv')

df['h'] = df['sigma_est'] * df['sigma_multiplier']

nlmeans = df[df['filter'] == 'nlmeans']
bm3d = df[df['filter'] == 'bm3d']

nlmeans_anscombe = nlmeans[nlmeans['anscombe_transform'] == True]
nlmeans_no_anscombe = nlmeans[nlmeans['anscombe_transform'] == False]
bm3d_anscombe = bm3d[bm3d['anscombe_transform'] == True]
bm3d_no_anscombe = bm3d[bm3d['anscombe_transform'] == False]

def plot_metrics_for_peak(peak, filter_name, df_no_anscombe, df_anscombe):
    plt.figure(figsize=(10, 6))

    # Gráficos de PSNR
    plt.subplot(2, 1, 1)
    plt.plot(df_no_anscombe['h'], df_no_anscombe['psnr_denoised'], label=f'{filter_name} sem Anscombe', marker='o')
    plt.plot(df_anscombe['h'], df_anscombe['psnr_denoised'], label=f'{filter_name} com Anscombe', marker='o')
    plt.xlabel('h')
    plt.ylabel('PSNR')
    plt.title(f'PSNR vs h ({filter_name}) - Peak: {peak}')
    plt.legend()
    plt.grid(True)

    # Gráficos de SSIM
    plt.subplot(2, 1, 2)
    plt.plot(df_no_anscombe['h'], df_no_anscombe['ssim_denoised'], label=f'{filter_name} sem Anscombe', marker='o')
    plt.plot(df_anscombe['h'], df_anscombe['ssim_denoised'], label=f'{filter_name} com Anscombe', marker='o')
    plt.xlabel('h')
    plt.ylabel('SSIM')
    plt.title(f'SSIM vs h ({filter_name}) - Peak: {peak}')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    

# cria gráficos para cada valor de 'peak'
for peak_value in df['peak'].unique():
    nlmeans_peak_no_anscombe = nlmeans_no_anscombe[nlmeans_no_anscombe['peak'] == peak_value]
    nlmeans_peak_anscombe = nlmeans_anscombe[nlmeans_anscombe['peak'] == peak_value]
    bm3d_peak_no_anscombe = bm3d_no_anscombe[bm3d_no_anscombe['peak'] == peak_value]
    bm3d_peak_anscombe = bm3d_anscombe[bm3d_anscombe['peak'] == peak_value]

    # Plotar os gráficos para NL-Means e BM3D para cada valor de peak
    plot_metrics_for_peak(peak_value, 'NL-Means', nlmeans_peak_no_anscombe, nlmeans_peak_anscombe)
    plot_metrics_for_peak(peak_value, 'BM3D', bm3d_peak_no_anscombe, bm3d_peak_anscombe)
