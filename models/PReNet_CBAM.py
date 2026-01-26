import torch
import torch.nn as nn
from .CBAM import CBAM
import torch.nn.functional as F

class PReNet_CBAM(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super().__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1)
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1)
        )
        self.res3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1)
        )
        self.res4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1)
        )
        self.res5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1)
        )

        # CBAM cho từng block
        self.cbam1 = CBAM(32)
        self.cbam2 = CBAM(32)
        self.cbam3 = CBAM(32)
        self.cbam4 = CBAM(32)
        self.cbam5 = CBAM(32)

        # ConvLSTM gates
        self.conv_i = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv_f = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv_g = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv_o = nn.Conv2d(64, 32, 3, 1, 1)

        self.conv = nn.Conv2d(32, 3, 3, 1, 1)

    def forward(self, input):
        b, _, h, w = input.size()
        x = input

        h_state = torch.zeros(b, 32, h, w, device=input.device)
        c_state = torch.zeros_like(h_state)

        for _ in range(self.iteration):
            x = torch.cat([input, x], dim=1)
            x = self.conv0(x)

            x_cat = torch.cat([x, h_state], dim=1)
            i = torch.sigmoid(self.conv_i(x_cat))
            f = torch.sigmoid(self.conv_f(x_cat))
            g = torch.tanh(self.conv_g(x_cat))
            o = torch.sigmoid(self.conv_o(x_cat))

            c_state = f * c_state + i * g
            h_state = o * torch.tanh(c_state)

            x = h_state

            res = x
            x = F.relu(self.cbam1(self.res1(x)) + res)
            res = x
            x = F.relu(self.cbam2(self.res2(x)) + res)
            res = x
            x = F.relu(self.cbam3(self.res3(x)) + res)
            res = x
            x = F.relu(self.cbam4(self.res4(x)) + res)
            res = x
            x = F.relu(self.cbam5(self.res5(x)) + res)

            x = self.conv(x)
            x = x + input

        return x

