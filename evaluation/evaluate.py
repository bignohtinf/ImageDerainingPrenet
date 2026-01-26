import torch
from tqdm import tqdm
from utils.image_utils import crop_border
from scripts.data.dataset import RainTestDataset
from .ssim import SSIMMetric
from torch.utils.data import DataLoader
from .psnr import calculate_psnr_batch

def evaluate(model, test_paths):
    model.eval()
    results = {}

    ssim_metric = SSIMMetric().cuda()

    with torch.no_grad():
        for dataset_name, paths in test_paths.items():
            print(f"\nEvaluating on {dataset_name}...")

            test_dataset = RainTestDataset(
                rain_dir=paths["rain"],
                gt_dir=paths["norain"]
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=2
            )

            psnr_total = 0
            ssim_total = 0

            for rain, gt in tqdm(test_loader, desc=dataset_name):
                rain, gt = rain.cuda(), gt.cuda()

                out, _ = model(rain)
                out = out.clamp(0, 1)

                # Crop border (theo paper)
                out = crop_border(out, shave=4)
                gt  = crop_border(gt, shave=4)

                psnr_total += calculate_psnr_batch(out, gt)
                ssim_total += ssim_metric(out, gt).item()

            psnr = psnr_total / len(test_loader)
            ssim = ssim_total / len(test_loader)

            results[dataset_name] = {
                "PSNR": psnr,
                "SSIM": ssim
            }

            print(f"[{dataset_name}] PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")

    return results