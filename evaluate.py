import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage import io, color
import os
import argparse
from scipy.stats import entropy

def calculate_sam(img1, img2):
    """Calcula el Spectral Angle Mapper (SAM) en grados."""
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    inner_product = np.sum(img1 * img2, axis=2)
    norm1 = np.sqrt(np.sum(img1**2, axis=2))
    norm2 = np.sqrt(np.sum(img2**2, axis=2))
    
    # Evitar divisiones por cero
    norm1[norm1 == 0] = 1e-6
    norm2[norm2 == 0] = 1e-6
    
    cos_theta = inner_product / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0) # Estabilidad numérica
    
    sam = np.arccos(cos_theta)
    return np.mean(sam) * (180 / np.pi)

def calculate_ergas(img1, img2):
    """Calcula el ERGAS (asumiendo misma resolución espacial)."""
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    # RMSE por canal
    rmse = np.sqrt(np.mean((img1 - img2)**2, axis=(0,1)))
    # Media de la imagen de referencia por canal
    mean_ref = np.mean(img2, axis=(0,1))
    mean_ref[mean_ref == 0] = 1e-6
    
    ergas = 100 * np.sqrt(np.mean((rmse / mean_ref)**2))
    return ergas

def rmetrics(corrected, reference):
    height, width, channels = reference.shape
    
    psnr = compare_psnr(corrected, reference)
    ssim = compare_ssim(corrected, reference, win_size=11, data_range=255, channel_axis=-1)
    sam = calculate_sam(corrected, reference)
    ergas = calculate_ergas(corrected, reference)

    clear_lab = color.rgb2lab(reference)
    dehaze_lab = color.rgb2lab(corrected)
    difference = color.deltaE_ciede2000(dehaze_lab, clear_lab)
    ciede = sum(sum(difference)) / height / width

    spatial_component = corrected[:, :, 0]             
    spectral_component = corrected[:, :, 1].flatten()  
    spatial_entropy = entropy(spatial_component)
    spectral_entropy = entropy(spectral_component)
    SSEQ_dehaze = spatial_entropy + spectral_entropy 

    spatial_component = reference[:, :, 0]             
    spectral_component = reference[:, :, 1].flatten()  
    spatial_entropy = entropy(spatial_component)
    spectral_entropy = entropy(spectral_component)
    SSEQ_clear = spatial_entropy + spectral_entropy   

    sseq = abs(sum(SSEQ_dehaze - SSEQ_clear) / sum(SSEQ_clear))
    if np.isinf(sseq) or np.isnan(sseq):
        sseq = 1
    
    return psnr, ssim, ciede, sseq, sam, ergas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-to", "--test_original", required=True, help="path to original testing images")
    ap.add_argument("-td", "--test_dehaze", required=True, help="path to dehazed testing images")
    args = vars(ap.parse_args())

    result_path = args["test_dehaze"]
    reference_path = args["test_original"]

    result_dirs = os.listdir(result_path)
    sum_psnr, sum_ssim, sum_ciede, sum_sseq, sum_sam, sum_ergas = 0., 0., 0., 0., 0., 0.

    cnt = 0

    for image_name in result_dirs:
        try: 
            corrected = io.imread(os.path.join(result_path, image_name))
            reference = io.imread(os.path.join(reference_path, image_name))
        except Exception:
            continue
            
        cnt += 1
        psnr, ssim, ciede, sseq, sam, ergas = rmetrics(corrected, reference)
        
        sum_psnr += psnr
        sum_ssim += ssim
        sum_ciede += ciede
        sum_sseq += sseq
        sum_sam += sam
        sum_ergas += ergas

        print(f'{image_name}: psnr={psnr:.4f} ssim={ssim:.4f} sam={sam:.4f} ergas={ergas:.4f} cie={ciede:.4f} sseq={sseq:.4f}')
        
        with open('metrics.txt', 'a') as f:
            f.write(f'{image_name}: psnr={psnr:.4f} ssim={ssim:.4f} sam={sam:.4f} ergas={ergas:.4f} cie={ciede:.4f} sseq={sseq:.4f}\n')

    mean_psnr = sum_psnr / cnt
    mean_ssim = sum_ssim / cnt
    mean_ciede = sum_ciede / cnt
    mean_sseq = sum_sseq / cnt
    mean_sam = sum_sam / cnt
    mean_ergas = sum_ergas / cnt
    
    res_str = f'Average: psnr={mean_psnr:.4f} ssim={mean_ssim:.4f} sam={mean_sam:.4f} ergas={mean_ergas:.4f} cie={mean_ciede:.4f} sseq={mean_sseq:.4f}\n'
    
    with open('metrics.txt', 'a') as f:
        f.write(res_str)
        
    print(res_str)

if __name__ == '__main__':
    main()