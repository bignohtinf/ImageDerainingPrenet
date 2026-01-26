from .ssim import SSIMMetric
from .psnr import calculate_psnr

class MetricCalculator:
    def __init__(self, max_val=1.0):
        self.max_val = max_val
        self.ssim_metric = SSIMMetric()

    def calculate_psnr(self, pred, target):
        return calculate_psnr(pred, target, self.max_val)

    def calculate_ssim(self, pred, target):
        if pred.is_cuda:
            self.ssim_metric = self.ssim_metric.cuda()
        return self.ssim_metric(pred, target).item()
