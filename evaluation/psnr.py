import torch

def calculate_psnr(self, pred, target):
    """Calculate PSNR"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    return 20 * torch.log10(self.max_val / torch.sqrt(mse)).item()
    
def calculate_psnr_batch(pred, target, max_val=1.0):
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)

    psnr = 0
    for i in range(pred.size(0)):
        mse = torch.mean((pred[i] - target[i]) ** 2)
        psnr += 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item() / pred.size(0)