import pandas as pd
import matplotlib.pyplot as plt

# Carregar dados do CSV
result_path = "data/results/ct
/"
csv_path = result_path + "2024-11-03.csv"

data = pd.read_csv(csv_path)

# Parâmetros de filtros e PSNR
filters = ["nlmeans", "bm3d"]
anscombe_transform_values = [True, False]
peak_values = [3000, 7500, 13000]

# Iterar sobre cada combinação
for filter_type in filters:
    for anscombe in anscombe_transform_values:
        for peak_value in peak_values:
            # Filtrar dados
            subset = data[(data["filter"] == filter_type) &
                          (data["anscombe_transform"] == anscombe) &
                          (data["peak"] == peak_value)]
            
            if not subset.empty:
                # Gráfico de PSNR Denoised x Sigma Multiplier
                plt.figure()
                plt.plot(subset["sigma_multiplier"], subset["psnr_denoised"], marker='o', label='PSNR Denoised')
                plt.xlabel("Sigma Multiplier")
                plt.ylabel("PSNR Denoised")
                plt.title(f"{filter_type.upper()} - Anscombe: {anscombe} - Peak: {peak_value}")
                plt.grid(True)
                if anscombe:
                    plt.savefig(f"{result_path}{filter_type}_ta_{peak_value}_psnr_denoised.png")
                else:
                    plt.savefig(f"{result_path}{filter_type}_{peak_value}_psnr_denoised.png")
                plt.close()

                # Gráfico de SSIM Denoised x Sigma Multiplier
                plt.figure()
                plt.plot(subset["sigma_multiplier"], subset["ssim_denoised"], marker='o', color='orange', label='SSIM Denoised')
                plt.xlabel("Sigma Multiplier")
                plt.ylabel("SSIM Denoised")
                plt.title(f"{filter_type.upper()} - Anscombe: {anscombe} - Peak: {peak_value}")
                plt.grid(True)
                if anscombe:
                    plt.savefig(f"{result_path}{filter_type}_ta_{peak_value}_ssim_denoised.png")
                else:
                    plt.savefig(f"{result_path}{filter_type}_{peak_value}_ssim_denoised.png")
                plt.close()