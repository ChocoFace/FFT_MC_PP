# ===============================
# FILE: nets_pytorch_kspace_2D.py
# ===============================
import numpy as np
import torch
from torch import nn, optim
import torch.fft
import tensorboardX
import time

# ---- Utilities ----
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ---- Simple bilinear RegularGridInterpolator in 2D (t, ky) ----
class RegularGridInterpolator2D:
    def __init__(self, points, values):
        # points = [w (Nt,), ky (Ny,)]
        self.w, self.ky = [p.contiguous() for p in points]
        self.values = values  # shape [Nt, Ny] complex
        assert self.values.ndim == 2

    def __call__(self, W, KY):
        # W, KY flattened (K,)
        idx_w_r = torch.bucketize(W, self.w).clamp(max=self.w.shape[0]-1)
        idx_w_l = (idx_w_r - 1).clamp(min=0)
        idx_k_r = torch.bucketize(KY, self.ky).clamp(max=self.ky.shape[0]-1)
        idx_k_l = (idx_k_r - 1).clamp(min=0)

        w_l = self.w[idx_w_l]; w_r = self.w[idx_w_r]
        ky_l = self.ky[idx_k_l]; ky_r = self.ky[idx_k_r]

        # avoid zero division
        dw = (w_r - w_l); dw[dw==0] = 1
        dk = (ky_r - ky_l); dk[dk==0] = 1
        tw = (W - w_l)/dw
        tk = (KY - ky_l)/dk

        # gather 4 neighbors
        v_ll = self.values[idx_w_l, idx_k_l]
        v_lr = self.values[idx_w_l, idx_k_r]
        v_rl = self.values[idx_w_r, idx_k_l]
        v_rr = self.values[idx_w_r, idx_k_r]

        # bilinear blend
        return (1-tw)*(1-tk)*v_ll + (1-tw)*tk*v_lr + tw*(1-tk)*v_rl + tw*tk*v_rr

# ---- Tiny 2D UNet-ish postprocess (minimal change) ----
class double_conv2d(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )

class UNetPP2d(nn.Module):
    def __init__(self, n_in=1, n_out=1, width=32):
        super().__init__()
        self.down1 = double_conv2d(n_in, width)
        self.down2 = double_conv2d(width, width*2)
        self.down3 = double_conv2d(width*2, width*4)
        self.down4 = double_conv2d(width*4, width*8)
        
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(width*8, width*4, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(width*4, width*2, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(width*2, width, 2, stride=2)
        
        self.conv_up3 = double_conv2d(width*8, width*4)
        self.conv_up2 = double_conv2d(width*4, width*2)
        self.conv_up1 = double_conv2d(width*2, width)
        self.out = nn.Conv2d(width, n_out, 1)
        self.step = nn.Parameter(torch.zeros(1,1,1,1))
        
    def forward(self, inp):
        c1 = self.down1(inp)   # x torch.Size([1, 1, 81, 256])
        x = self.pool(c1)
        c2 = self.down2(x)
        x = self.pool(c2)
        c3 = self.down3(x)
        x = self.pool(c3)
        c4 = self.down4(x)
        
        x = self.up3(c4)
        x = torch.cat([x, c3], dim=1)
        x = self.conv_up3(x)
        x = self.up2(x)
        x = torch.cat([x, c2], dim=1)
        x = self.conv_up2(x)
        x = self.up1(x)
        x = torch.cat([x, c1], dim=1)
        x = self.conv_up1(x)
        upd = self.out(x)
        # return x[:, :1] * 0 + self.step * upd  # minimal-change: residual gate
        return inp + self.step * upd  # minimal-change: residual gate

# ---- Inverse operators (2D) ----
class PATinv2D(nn.Module):
    def __init__(self, sf, points_inv, pointsI_inv, c, NtFactor, NyI):
        super().__init__()
        self.sf = sf              # [1,1,Nt,Ny]
        self.points_inv = points_inv   # [w_flat, ky_flat]
        self.pointsI_inv = pointsI_inv # [wI_flat, kyI_flat]
        self.c = c
        self.NtFactor = NtFactor
        self.NyI = NyI

    def forward(self, pT):
        # pT: [B,1,Nt,Ny]
        B = pT.shape[0]
        # time-symmetric pad
        pUp = torch.flip(pT[:, :, :, :], dims=[2])  # flip time axis
        p0 = torch.cat([pUp[:, :, None, 0, :], pT[:, :, None, :, :][:, :, :, 1:, :]], dim=2).squeeze(2)
        # FFT (2D: time x y)
        p0 = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(p0, dim=(2,3)), dim=(2,3)), dim=(2,3))
        p0 = p0 * self.sf  # spectral filter
        p0 = p0[0,0]       # [Nt, Ny] complex (batch=1 assumed in training loop)
        gi = RegularGridInterpolator2D(self.points_inv, p0)
        p0I = gi(self.pointsI_inv[0], self.pointsI_inv[1])
        p0I = p0I.reshape(1, 1, self.NyI[0], self.NyI[1])
        # iFFT back to (t,y)
        p0I = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(p0I, dim=(2,3)), dim=(2,3)), dim=(2,3)).real
        # slice valid time window
        T = p0I.shape[2]
        indS = int(torch.ceil((torch.tensor(T)+1)/2/self.NtFactor).item())
        indE = int(torch.floor(torch.tensor(T)/self.NtFactor).item()) + 1
        p0I = 4 * p0I[:, :, indS:indE, :] / self.c / self.NtFactor
        return p0I

class PATinvNN2D(nn.Module):
    def __init__(self, sf, points_inv, pointsI_inv, c, NtFactor, NyI, Unet):
        super().__init__()
        self.op = PATinv2D(sf, points_inv, pointsI_inv, c, NtFactor, NyI)
        self.unet = Unet
    def forward(self, pT):
        p0 = self.op(pT)
        p0I = self.unet(p0)
        return p0, p0I

# ---- Post-process wrappers (2D) ----
class postProcess2D(nn.Module):
    def __init__(self, op_adj, unet):
        super().__init__()
        self.op = op_adj
        self.unet = unet
    def forward(self, pT):
        p0 = self.op(pT)
        p0u = self.unet(p0)
        return p0, p0u

# ---- Training loops (2D) ----
def train_postprocess_2d(dataSet, sfi, ky, w, kyI, wI, c, NtFactor, expName, filePath,
                         netType='unet', lossFunc='l2', bSize=1, trainIter=10000,
                         useTensorboard=True, lValInit=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if useTensorboard:
        tr = tensorboardX.SummaryWriter(logdir=filePath + f"/runs/{expName}/train/")
        te = tensorboardX.SummaryWriter(logdir=filePath + f"/runs/{expName}/test/")
    sfi = torch.from_numpy(sfi)[None, None].to(device)  # [1,1,Nt,Ny]
    points_inv = [torch.from_numpy(w.flatten()).to(device), torch.from_numpy(ky.flatten()).to(device)]
    pointsI_inv = [torch.from_numpy(wI.flatten()).to(device), torch.from_numpy(kyI.flatten()).to(device)]
    NyIm = torch.tensor(kyI.shape, device=device)
    op = PATinv2D(sfi, points_inv, pointsI_inv, torch.tensor(c, device=device), torch.tensor(NtFactor, device=device), NyIm).to(device)
    if netType == 'unet':
        unet = UNetPP2d(1,1,32).to(device)
    else:
        unet = UNetPP2d(1,1,32).to(device)
    model = postProcess2D(op, unet).to(device)
    criterion = nn.MSELoss(); optimizer = optim.Adam(model.parameters(), lr=lValInit)
    sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, trainIter)
    print('Params:', count_parameters(model))
    for it in range(trainIter):
        sched.step()
        images = torch.from_numpy(dataSet.train.next_batch(bSize)[1]).float().to(device)  # [B,1,Ny,Nx]
        projs = torch.from_numpy(dataSet.train.next_batch(bSize)[0]).float().to(device)   # [B,1,Nt,Ny]
        optimizer.zero_grad()
        p00, p0u = model(projs)
        loss = criterion(p0u, images)
        loss.backward(); optimizer.step()
        if useTensorboard and it % 25 == 0:
            tr.add_scalar('loss', loss.item(), it)
    torch.save(model, filePath + expName + '.pt')
    return model