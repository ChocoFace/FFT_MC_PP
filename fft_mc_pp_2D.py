# ===============================
# FILE: fft_mc_pp_2D.py  (FFT + Model Correction + Post-Process)
# ===============================
import math
import torch
from torch import nn, optim
import torch.fft
import time
import h5py
import numpy as np
from kgrids_pytorch_kspace_2D import kspaceKGrids2D
from nets_pytorch_kspace_2D import RegularGridInterpolator2D, UNetPP2d
from complex_layers_2D import ComplexConv2d, NaiveComplexBatchNorm2d, ComplexReLU, ComplexConvTranspose2d

# ---- Complex UNet in k-space (acts before iFFT) ----
class complex_double_conv2d(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            ComplexConv2d(in_c, out_c, 3, padding=1),
            NaiveComplexBatchNorm2d(out_c),
            ComplexReLU(),
            ComplexConv2d(out_c, out_c, 3, padding=1),
            NaiveComplexBatchNorm2d(out_c),
            ComplexReLU(),
        )
    def forward(self, z): return self.net(z)

class ComplexUNet2D(nn.Module):
    def __init__(self, in_c=1, out_c=1, width=32):
        super().__init__()
        self.d1 = complex_double_conv2d(in_c, width)
        self.d2 = complex_double_conv2d(width, width*2)
        self.d3 = complex_double_conv2d(width*2, width*4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = ComplexConvTranspose2d(width*4, width*2, 2, stride=2)
        self.up1 = ComplexConvTranspose2d(width*2, width, 2, stride=2)
        self.u2 = complex_double_conv2d(width*4, width*2)
        self.u1 = complex_double_conv2d(width*2, width)
        self.out = ComplexConv2d(width, out_c, 1, padding=0)
        self.step = nn.Parameter(torch.zeros(1))
    def forward(self, z):
        c1 = self.d1(z)
        x = self.pool(torch.view_as_real(c1).permute(0,3,1,2))  # pool on magnitude-like tensor
        x = torch.view_as_complex(x.permute(0,2,3,1).contiguous())
        c2 = self.d2(x)
        x = self.pool(torch.view_as_real(c2).permute(0,3,1,2))
        x = torch.view_as_complex(x.permute(0,2,3,1).contiguous())
        c3 = self.d3(x)
        x = self.up2(c3)
        x = torch.cat([x, c2], dim=1)
        x = self.u2(x)
        x = self.up1(x)
        x = torch.cat([x, c1], dim=1)
        x = self.u1(x)
        upd = self.out(x)
        return self.step * upd

# ---- PAT inverse with MC in k-space + PP in image space ----
class PATinvNN_PP_2D(nn.Module):
    def __init__(self, sf, points_inv, pointsI_inv, c, NtFactor, NyI, mc_unet, pp_unet):
        super().__init__()
        self.sf = sf
        self.points_inv = points_inv
        self.pointsI_inv = pointsI_inv
        self.c = c; self.NtFactor = NtFactor; self.NyI = NyI
        self.mc_unet = mc_unet      # complex UNet in (w, ky)
        self.pp_unet = pp_unet      # real UNet in (t, y) after iFFT

    def forward(self, pT):
        # pT: [B,1,Nt,Ny] real
        # ---- FFT to (w, ky)
        pUp = torch.flipud(pT[0,0,:,:]) #torch.Size([2029, 256])
        p0 = torch.cat((pUp[None,None,:,:],pT[:,:,1::,:]),2)  #torch.Size([1, 1, 81, 256])
        indS = math.ceil((p0.shape[2]+1)/2/self.NtFactor)
        indE = math.floor((p0.shape[2])/self.NtFactor)+1
        F = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(pT, dim=(2,3)), dim=(2,3)), dim=(2,3))
        F = F * self.sf  # physics filter
        # Interp to target grid (flatten then reshape)
        F0 = F[0,0]  # assume B=1 for training/eval
        gi = RegularGridInterpolator2D(self.points_inv, F0)
        Fin = gi(self.pointsI_inv[0], self.pointsI_inv[1]).reshape(1,1,self.NyI[0], self.NyI[1])
        # ---- Model correction in k-space (complex)
        Fmc = self.mc_unet(Fin)
        # ---- iFFT back to (t,y)
        f_fft = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(Fin, dim=(2,3)), dim=(2,3)), dim=(2,3)).real
        f_mc  = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(Fmc, dim=(2,3)), dim=(2,3)), dim=(2,3)).real
        # slice time window
        
        p0     = 4 * f_fft[:,:,408:664,:] / self.c / self.NtFactor
        p0_mc  = 4 * f_mc [:,:,408:664,:] / self.c / self.NtFactor
        # ---- post-processing in image domain
        p0_pp = self.pp_unet(p0_mc)
        return p0, p0_mc, p0_pp


def train_fft_mc_pp_2d(dataset, sfi, ky, w, kyI, wI, c, NtFactor, expName, filePath,
                        bSize=1, trainIter=50000, lr=1e-2, useTensorboard=True,
                        h5_path=None, x_key='X_trn', y_key='Y_trn'):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if useTensorboard:
        from tensorboardX import SummaryWriter
        tr = SummaryWriter(filePath + f'/runs/{expName}/train')
        te = SummaryWriter(filePath + f'/runs/{expName}/test')
    sfi = torch.from_numpy(sfi)[None,None].to(device)
    points_inv = [torch.from_numpy(w.flatten()).to(device), torch.from_numpy(ky.flatten()).to(device)]
    pointsI_inv= [torch.from_numpy(wI.flatten()).to(device), torch.from_numpy(kyI.flatten()).to(device)]
    NyI = torch.tensor(kyI.shape, device=device)
    mc = ComplexUNet2D(1,1,32).to(device)
    pp = UNetPP2d(1,1,32).to(device)
    model = PATinvNN_PP_2D(sfi, points_inv, pointsI_inv, torch.tensor(c, device=device), torch.tensor(NtFactor, device=device), NyI, mc, pp).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, trainIter)
    crit = nn.MSELoss()
    print('Params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    #---------------------------------------------------------------------------------------------------
    # -------------------------------------------- Data sources ----
    #---------------------------------------------------------------------------------------------------
    # use_h5 = h5_path is not None
    # if use_h5:
    #     f = h5py.File(h5_path, 'r')
    #     Xds = f[x_key]  # shape [N,1,Nt,Ny]
    #     Yds = f[y_key]  # shape [N,1,Ny,Nx]
    #     N = Xds.shape[0]
    #     if bSize != 1:
    #         print('[WARN] 当前实现要求 bSize==1（k-space 插值使用单样本）。已强制使用 batch=1。')
    #         bSize = 1
    #     order = np.random.permutation(N)
    #     ptr = 0
    #     def next_batch_h5(B):
    #         nonlocal ptr, order
    #         if ptr + B > N:
    #             order = np.random.permutation(N)
    #             ptr = 0
    #         idx = order[ptr:ptr+B]
    #         ptr += B
    #         x = np.asarray(Xds[idx])  # (B,1,Nt,Ny)
    #         y = np.asarray(Yds[idx])  # (B,1,Ny,Nx)
    #         return x, y
    # else:
    #     if not hasattr(dataset, 'train') or not hasattr(dataset.train, 'next_batch'):
    #         raise ValueError('Provide either `dataset.train.next_batch` or `h5_path`.')
    #---------------------------------------------------------------------------------------------------
    for it in range(trainIter):
        sched.step(); opt.zero_grad()
        x, y = dataset.train.next_batch(bSize)
        projs = torch.from_numpy(x).float().to(device)
        imgs  = torch.from_numpy(y).float().to(device)
        p0, p0mc, p0pp = model(projs)
        loss = crit(p0pp, imgs)
        loss.backward(); opt.step()
        if useTensorboard and it % 25 == 0:
            tr.add_scalar('loss', loss.item(), it)
    torch.save(model, filePath + expName + '.pt')
    return model


def eval_fft_mc_pp_2d(dataset, expName, filePath, saveResults=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(filePath + expName + '.pt', map_location=device)
    inds = dataset.test.ind
    X, Y = dataset.test.selected_set(inds)
    X = torch.from_numpy(X).float().to(device)
    res = []
    for i in range(X.shape[0]):
        x = X[i:i+1]
        with torch.no_grad():
            p0, p0mc, p0pp = model(x)
        for name, arr in [('p0',p0), ('p0MC',p0mc), ('p0MCPP',p0pp)]:
            npy = arr.cpu().numpy()
            if saveResults:
                np.save(filePath + f'{expName}_{name}_{inds[i]}', npy)
        res.append((p0.cpu().numpy(), p0mc.cpu().numpy(), p0pp.cpu().numpy()))
    if saveResults:
        np.save(filePath + f'{expName}_ind', inds)
    return res