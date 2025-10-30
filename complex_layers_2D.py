# ===============================
# FILE: complex_layers_2D.py (minimal complex ops via 2-channel real convs)
# ===============================
import torch
from torch import nn

class ComplexToChannels(nn.Module):
    def forward(self, z: torch.Tensor):
        # [..., H, W] complex -> [..., 2, H, W] real
        return torch.stack([z.real, z.imag], dim=1)

class ChannelsToComplex(nn.Module):
    def forward(self, x: torch.Tensor):
        # [..., 2, H, W] -> complex [..., H, W]
        return x[:,0] + 1j*x[:,1]

class ComplexConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding=0, bias=True):
        super().__init__()
        # treat complex channel as 2 real channels
        self.conv = nn.Conv2d(in_ch*2, out_ch*2, kernel_size, padding=padding, bias=bias)
    def forward(self, z):
        # z: [B,1,H,W] complex mapped as two real channels
        zr = torch.cat([z.real, z.imag], dim=1)  # [B,2,H,W]
        y = self.conv(zr)
        return y[:,0] + 1j*y[:,1]

class NaiveComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features*2)
    def forward(self, z):
        zr = torch.stack([z.real, z.imag], dim=1)
        y = self.bn(zr)
        return y[:,0] + 1j*y[:,1]

class ComplexReLU(nn.Module):
    def forward(self, z):
        return torch.relu(z.real) + 1j*torch.relu(z.imag)

class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch*2, out_ch*2, kernel_size, stride=stride, padding=padding)
    def forward(self, z):
        zr = torch.stack([z.real, z.imag], dim=1)
        y = self.deconv(zr)
        return y[:,0] + 1j*y[:,1]