import math
import h5py
import numpy as np
from kgrids_pytorch_kspace_2D import kspaceKGrids2D
from show_img import show_three_images,show_one_images
import torch
from torch import nn, optim
import torch.fft
import time
from kgrids_pytorch_kspace_2D import kspaceKGrids2D
from nets_pytorch_kspace_2D import RegularGridInterpolator2D, UNetPP2d
# from complex_layers_2D import ComplexConv2d, NaiveComplexBatchNorm2d, ComplexReLU, ComplexConvTranspose2d
from complexPyTorch.complexLayers import ComplexConv2d,NaiveComplexBatchNorm2d,ComplexReLU, ComplexConvTranspose2d
# from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d

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

class resNetComplex(nn.Module):
    def __init__(self,n_in,n_out, width_channels):
        super().__init__()
        self.doubleConv = nn.Sequential(
           ComplexConv2d(n_in, width_channels, 3, padding=1),
           NaiveComplexBatchNorm2d(width_channels),   
           ComplexReLU(),
           ComplexConv2d(width_channels, width_channels, 3, padding=1),
           NaiveComplexBatchNorm2d(width_channels),   
           ComplexReLU(),
           ComplexConv2d(width_channels, n_out, 3, padding=1,bias=False)
       )
        
    def forward(self, cur):
        
        update = self.doubleConv(cur)
        
        return update

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
        pT = pT.to(dtype=torch.complex64).unsqueeze(0).unsqueeze(0) #torch.Size([1, 1, 2029, 256])
        pUp = torch.flipud(pT[0,0,:,:]) #torch.Size([2029, 256])
        p0 = torch.cat((pUp[None,None,:,:],pT[:,:,1::,:]),2)  #torch.Size([1, 1, 81, 256])
        indS = math.ceil((p0.shape[2]+1)/2/self.NtFactor)
        indE = math.floor((p0.shape[2])/self.NtFactor)+1
        F = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(p0, dim=(2,3)), dim=(2,3)), dim=(2,3))
        F = F * self.sf  # physics filter
        # Interp to target grid (flatten then reshape)
        F0 = F[0,0]  # assume B=1 for training/eval
        gi = RegularGridInterpolator2D(self.points_inv, F0)
        Fin = gi(self.pointsI_inv[0], self.pointsI_inv[1]).reshape(1,1,self.NyI[0], self.NyI[1])
        Fin = Fin.to(dtype=torch.complex64)
        # ---- Model correction in k-space (complex)
        Fmc = Fin + self.mc_unet(Fin)
        # ---- iFFT back to (t,y)
        f_fft = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(Fin, dim=(2,3)), dim=(2,3)), dim=(2,3)).real
        f_mc  = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(Fmc, dim=(2,3)), dim=(2,3)), dim=(2,3)).real
        f_0 = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(F, dim=(2,3)), dim=(2,3)), dim=(2,3)).real
        # slice time window

        p0     = 4 * f_fft[:,:,indS:indE,:] / self.c / self.NtFactor
        p0_mc  = 4 * f_mc [:,:,indS:indE,:] / self.c / self.NtFactor
        # ---- post-processing in image domain
        p0_mc = p0_mc[:,:,2:258,:]
        p0_pp = self.pp_unet(p0_mc)
        return p0, p0_mc, p0_pp
    
h5_path = "/root/data-fs/yueg_data/Data_skin_h5/data_cache.h5"
use_h5 = h5_path is not None
x_key='X_trn'
y_key='Y_trn'
bSize = 1  # batch size
if use_h5:
    f = h5py.File(h5_path, 'r')
    Xds = f[x_key]  # shape [N,1,Nt,Ny]
    Yds = f[y_key]  # shape [N,1,Ny,Nx]
    N = Xds.shape[0]
    if bSize != 1:
        print('[WARN] 当前实现要求 bSize==1（k-space 插值使用单样本）。已强制使用 batch=1。')
        bSize = 1
order = np.random.permutation(N)
ptr = 0
def next_batch_h5(B):
    global ptr, order
    if ptr + B > N:
        order = np.random.permutation(N)
        ptr = 0
    idx = order[ptr:ptr+B]
    ptr += B
    x = np.asarray(Xds[idx])  # (B,1,Nt,Ny)
    y = np.asarray(Yds[idx])  # (B,1,Ny,Nx)
    return x, y
for it in range(5):
    # xb, yb = next_batch_h5(bSize)
    import scipy.io as io
    xb = io.loadmat("/root/data-fs/yueg_data/Data_skin_correct_distrib/Sig/val/sinograms/45model.mat")['test_sig1']
    yb = io.loadmat("/root/data-fs/yueg_data/Data_skin_correct_distrib/Initial/Img_trn_val/45model.mat")['initial']
    # dx=5.3e-5*2; dy=5.3e-5*2; dt=3*5/3*1e-8; c=1500; NtFactor=5
    dx=0.106e-3; dy=0.106e-3; dt=1/62.5e6; c=1500; NtFactor=5


    # 用一条测试样本构建 k-space 插值网格（p(t,y)）
    # sample_pTy = xb[0,0]  # [Nt, Ny]
    sample_pTy = xb # [Nt, Ny]
    kg = kspaceKGrids2D(dx,dy,dt,c,NtFactor)
    sfi, ky, w, kyI, wI = kg.inverse(sample_pTy.astype(np.float32))
    # show_three_images(np.abs(np.squeeze(sfi)), np.abs(np.squeeze(wI)), np.abs(np.squeeze(kyI)), 'kgrids_it'+str(it))
    # print(f'it={it}: xb.shape={xb.shape}, yb.shape={yb.shape}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sfi = torch.from_numpy(sfi)[None,None].to(device)
    points_inv = [torch.from_numpy(w.flatten()).to(device), torch.from_numpy(ky.flatten()).to(device)]
    pointsI_inv= [torch.from_numpy(wI.flatten()).to(device), torch.from_numpy(kyI.flatten()).to(device)]
    NyI = torch.tensor(kyI.shape, device=device)
    print(f'sfi={sfi[0].shape}: points_inv.shape={points_inv[0].shape}, pointsI_inv.shape={pointsI_inv[0].shape}')
    # mc = ComplexUNet2D(1,1,32).to(device)
    mc = resNetComplex(1,1,32).to(device)
    pp = UNetPP2d(1,1,32).to(device)
    model = PATinvNN_PP_2D(sfi, points_inv, pointsI_inv, torch.tensor(c, device=device), torch.tensor(NtFactor, device=device), NyI, mc, pp).to(device)
    
    opt = optim.Adam(model.parameters(), lr=0.001)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, 200)
    # sched = optim.lr_scheduler.CosineAnnealingLR(opt, trainIter)
    crit = nn.MSELoss()
    print('Params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    for it in range(5):
        # sched.step()
        opt.zero_grad()
        # x, y = next_batch_h5(bSize)
        x = io.loadmat("/root/data-fs/yueg_data/Data_skin_correct_distrib/Sig/val/sinograms/45model.mat")['test_sig1']
        y = io.loadmat("/root/data-fs/yueg_data/Data_skin_correct_distrib/Initial/Img_trn_val/45model.mat")['initial']
        projs = torch.from_numpy(x).float().to(device)
        imgs  = torch.from_numpy(y).float().to(device)
        p0, p0mc, p0pp = model(projs)
