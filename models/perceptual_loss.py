import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_ids=[3, 8, 15], use_gpu=True):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        self.layers = nn.ModuleList()
        prev = 0
        for lid in layer_ids:
            self.layers.append(vgg[prev:lid])
            prev = lid

        for p in self.parameters():
            p.requires_grad = False

        if use_gpu:
            self.cuda()

    def forward(self, x, y):
        loss = 0
        for layer in self.layers:
            x = layer(x)
            y = layer(y)
            loss += F.l1_loss(x, y)
        return loss
