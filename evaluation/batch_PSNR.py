import numpy as np
try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
except:
    from skimage.measure import compare_psnr as peak_signal_noise_ratio
    from skimage.measure import compare_ssim as structural_similarity

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return (PSNR/Img.shape[0])