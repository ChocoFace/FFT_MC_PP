import math
import numpy as np
from scipy import io
import torch
import numpy as np
from nets_pytorch_kspace_2D import RegularGridInterpolator2D


def pat_fft_linear_v3(p_xt, dt, dx, c, z_max, Nz, time_tukey_alpha=0.2, space_hann=True, eps=1e-9):
    """
    线阵 PAT 的 FFT/k-space (Stolt) 重建（稳定版）
    - 轴顺序: 输入应为 [Nt, Nx]；若检测到 [Nx, Nt] 会自动转置
    - 频域顺序: 全程“自然序”，不做任何 fftshift/ifftshift
      * kx: np.fft.fftfreq -> [0, +, ..., -, ...]
      * kz: rfft 的自然序 [0..kmax]
    - kz 维用 irfft，避免手工补负半轴导致的条纹
    """
    # ---- 形状自检（时间在前）----
    Nt0, Nx0 = p_xt.shape
    if Nx0 > Nt0:   # 很多数据存成 [Nx, Nt]
        p_xt = p_xt.T
    Nt, Nx = p_xt.shape

    # ---- 窗函数（抑旁瓣/直流）----
    if time_tukey_alpha > 0:
        n = np.arange(Nt)
        w = np.ones(Nt)
        edge = int(np.floor(time_tukey_alpha*(Nt-1)/2))
        if edge > 0:
            ramp = 0.5*(1 + np.cos(np.pi*(2*n[:edge]/(time_tukey_alpha*(Nt-1)) - 1)))
            w[:edge] = ramp
            w[-edge:] = ramp[::-1]
        p_xt = (w[:, None]) * p_xt
    if space_hann:
        wx = 0.5 - 0.5*np.cos(2*np.pi*np.arange(Nx)/Nx)
        p_xt = p_xt * wx[None, :]

    # ---- 到 (kx, ω) 域（自然序）----
    P_wx  = np.fft.rfft(p_xt, axis=0)                     # (Nw, Nx)
    omega = 2*np.pi*np.fft.rfftfreq(Nt, d=dt)             # (Nw,)
    P_w_kx = np.fft.fft(P_wx, axis=1)                     # (Nw, Nx) 自然序
    kx = 2*np.pi*np.fft.fftfreq(Nx, d=dx)                 # (Nx,) 自然序

    # ---- 目标 kz 网格（单边，含 0；自然序）----
    dkz = 2*np.pi / z_max
    kz_max_by_bw = omega.max() / c
    Nz_req = int(np.floor(kz_max_by_bw / dkz)) + 1
    Nz = int(min(Nz, Nz_req))
    kz_grid = np.arange(Nz) * dkz                         # 0..(Nz-1)*dkz

    # ---- Stolt: 对每个 kx，沿 ω 做 1D 线性插值到 ω* = c*sqrt(kx^2 + kz^2) ----
    Fpos = np.zeros((Nx, Nz), dtype=np.complex128)        # kx(自然序) × kz(非负)
    domega = omega[1] - omega[0]

    for ix, kxi in enumerate(kx):
        omega_need = c*np.sqrt(kxi*kxi + kz_grid*kz_grid)  # (Nz,)
        mask = (omega_need <= omega[-1])
        if not np.any(mask):
            continue

        idx   = omega_need[mask] / domega
        i0    = np.floor(idx).astype(int)
        alpha = idx - i0
        i0    = np.clip(i0, 0, len(omega)-2)
        i1    = i0 + 1

        Pw = P_w_kx[:, ix]                                 # (Nw,)
        samp = (1-alpha)*Pw[i0] + alpha*Pw[i1]

        kz_sel    = kz_grid[mask]
        omega_sel = omega_need[mask]
        kz_safe   = np.maximum(kz_sel, eps)

        # 预加权（与 UBP 等价）：2 i ω / (c^2 kz)
        scale = (2j*omega_sel) / (c*c * kz_safe)
        Fpos[ix, mask] = scale * samp

    # ---- kz 维直接 irfft（自动补负半轴），长度设为 N = 2*(Nz-1) ----
    Nz_full = max(2*(Nz-1), 2)                             # 保证>=2且为偶数
    img_zx = np.fft.irfft(Fpos, n=Nz_full, axis=1)         # -> (Nx, Nz_full)，实值

    # ---- kx 维 ifft 回到 x ----
    img_zx = np.fft.ifft(img_zx, axis=0)                   # -> (Nx, Nz_full)
    img_zx = np.real(img_zx).T                             # -> (Nz_full, Nx)，(z,x)

    # ---- 生成坐标轴；只取 z>=0 的前 Nz 片，刚好覆盖 ~ z_max ----
    dz = 2*np.pi / (Nz_full * dkz)
    z = np.arange(Nz) * dz
    x = np.arange(Nx) * dx
    img = img_zx[:Nz, :]

    return img, x, z

def pat_ubp_linear(p_xt, dt, dx, c, z_max, Nz):
    """
    线阵 UBP（2D：x-z平面，传感器沿x、位于z=0）
    p_xt: [Nt, Nx]（若传来的是 [Nx,Nt] 会自动转置）
    返回 img(z,x), x(m), z(m)
    """
    Nt0, Nx0 = p_xt.shape
    if Nx0 > Nt0:  # 很多数据存成 [Nx,Nt]
        p_xt = p_xt.T
    Nt, Nx = p_xt.shape

    # 轴
    x = (np.arange(Nx) - (Nx-1)/2) * dx
    z = np.linspace(0.0, z_max, Nz)
    t = np.arange(Nt) * dt

    # 预处理：去直流 & 时间窗（抑旁瓣）
    p = p_xt - p_xt.mean(axis=0, keepdims=True)
    win = np.hanning(Nt)[:,None]
    p = p * win

    # 对 t 做一次时间微分（UBP权重里等效有 ∂/∂t）
    dpdt = np.gradient(p, dt, axis=0)

    # delay-and-sum（矢量化线性插值）
    Xs = x[None, :]                  # (1,Nx)
    X  = x[None, None, :]            # (1,1,Nx)
    Z  = z[:, None, None]            # (Nz,1,1)
    R  = np.sqrt((X - Xs)**2 + Z**2) # (Nz,1,Nx)
    tau = R / c                      # (Nz,1,Nx)
    u = tau / dt                     # 连续时间索引
    i0 = np.clip(np.floor(u).astype(int), 0, Nt-2)
    a  = (u - i0)                    # 线性插值权重

    # 取样并累加（带 1/R 权重，经验上抑制近场过亮）
    s0 = dpdt[i0, np.arange(Nx)]     # 广播：(Nz,1,Nx)->(Nz,Nx)
    s1 = dpdt[i0+1, np.arange(Nx)]
    samp = (1-a)*s0 + a*s1
    img = (samp / np.maximum(R.squeeze(1), 1e-9)).sum(axis=1)  # (Nz,)

    # 结果 (Nz, Nx)
    return img, x, z
def fft(xb):
    pT=torch.from_numpy(xb).unsqueeze(0).unsqueeze(0)  # (1,1,Nt,Ny)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dx=0.106e-3; dy=0.106e-3; dt=1/62.5e6; c=1500; NtFactor=5
    sample_pTy = xb # [Nt, Ny]
    kg = kspaceKGrids2D(dx,dy,dt,c,NtFactor)
    sfi, ky, w, kyI, wI = kg.inverse(sample_pTy.astype(np.float32))
    sfi = torch.from_numpy(sfi)[None,None].to(device)
    points_inv = [torch.from_numpy(w.flatten()).to(device), torch.from_numpy(ky.flatten()).to(device)]
    pointsI_inv= [torch.from_numpy(wI.flatten()).to(device), torch.from_numpy(kyI.flatten()).to(device)]
    NyI = torch.tensor(kyI.shape, device=device)
    print(f'sfi={sfi[0].shape}: points_inv.shape={points_inv[0].shape}, pointsI_inv.shape={pointsI_inv[0].shape}')
    pT = pT.to(dtype=torch.complex64)
    pUp = torch.flipud(pT[0,0,:,:])
    p0 = torch.cat((pUp[None,None,:,:],pT[:,:,1::,:]),2)  #torch.Size([1, 1, 4057, 256])
    indS = math.ceil((p0.shape[2]+1)/2/NtFactor)
    indE = math.floor((p0.shape[2])/NtFactor)+1
    F = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(p0, dim=(2,3)), dim=(2,3)), dim=(2,3))
    F = F * sfi # physics filter
    # Interp to target grid (flatten then reshape)
    F0 = F[0,0]  # assume B=1 for training/eval
    gi = RegularGridInterpolator2D(points_inv, F0)
    Fin = gi(pointsI_inv[0], pointsI_inv[1]).reshape(1,1,NyI[0], NyI[1])
    Fin = Fin.to(dtype=torch.complex64)

    # ---- iFFT back to (t,y)
    f_fft = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(Fin, dim=(2,3)), dim=(2,3)), dim=(2,3)).real
    f_0 = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(F, dim=(2,3)), dim=(2,3)), dim=(2,3)).real
    p0     = 4 * f_fft[:,:,indS:indE,:] / c / NtFactor
    return f_fft,p0 
    
class kspaceKGrids2D(object):
    """2D version of k-space grids. Drop kz / z-dimension entirely.
    Assumes data True: imagesTrue(y, x, sample), dataTrue(sensor_y, time, sample)
    """
    def __init__(self, dx, dy, dt, c, NtFactor):
        self._dx = dx
        self._dy = dy
        self._dt = dt
        self._c = c
        self._NtFactor = NtFactor

    @property
    def dx(self):
        return self._dx

    @property
    def dy(self):
        return self._dy

    @property
    def dt(self):
        return self._dt

    @property
    def c(self):
        return self._c

    @property
    def NtFactor(self):
        return self._NtFactor

    def inverse(self, pT):
        """
        Inputs (2D case):
            pT: pressure time series with shape [Nt, Ny]
        Output k-space grids & scalar factors analogous to 3D version, but 2D.
        Returns: sf, ky, w, kyI, wI
        """
        # time-mirror and pad (same idea as 3D but without z)
        pT = np.concatenate((pT[::-1, :], pT[1:, :]), axis=0)  #(4057, 256)
        Nt, Ny = pT.shape  # 4057, 256

        # build k-grids (target / interp)
        kgridBack = kgrid2D(math.ceil(Nt / self.NtFactor), self.dx, Ny, self.dy, self.c)
        kgridBackC = kgrid2D(Nt, self.dx, Ny, self.dy, self.dx / self.dt)

        c = kgridBackC.c
        w = c * kgridBackC.kx   # temporal frequency axis
        w_new = kgridBack.c * kgridBack.k

        # 2D dispersion (no kz term)
        sf = (np.square(w / c) - np.square(kgridBackC.ky))
        sf = c * c * np.sqrt(sf.astype(np.complex64))
        sf = np.divide(sf, 2 * w, out=np.zeros_like(sf), where=(w != 0))

        # handle singularities / evanescent removal like 3D
        idx_center = np.where((w == 0) & (kgridBackC.ky == 0))
        if idx_center[0].size > 0:
            sf[idx_center] = c / 2
        idx_evan = np.where(np.abs(w) < c * np.abs(kgridBackC.ky))
        sf[idx_evan] = 0
        sf[np.isnan(sf)] = 0

        ky = kgridBackC.ky_vec
        kyI = kgridBack.ky
        w = c * kgridBackC.kx_vec
        wI = w_new

        return sf, ky, w, kyI, wI
    
class kgrid2D(object):
    def __init__(self, Nx, dx, Ny, dy, c):
        k, kx, ky, kx_vec, ky_vec = makeKgrid2D(Nx, dx, Ny, dy)
        self._kx = kx
        self._ky = ky
        self._k = k
        self._kx_vec = kx_vec
        self._ky_vec = ky_vec
        self.Nx = Nx; self.Ny = Ny
        self.dx = dx; self.dy = dy
        self.c = c

    @property
    def kx(self): return self._kx
    @property
    def ky(self): return self._ky
    @property
    def k(self): return self._k
    @property
    def kx_vec(self): return self._kx_vec
    @property
    def ky_vec(self): return self._ky_vec


def makeKgrid2D(Nx, dx, Ny, dy):
    kx_vec = makeDim(Nx, dx)
    ky_vec = makeDim(Ny, dy)
    kx = np.tile(kx_vec.reshape(-1, 1), (1, Ny))
    ky = np.tile(ky_vec.reshape(1, -1), (Nx, 1))
    k = np.sqrt(kx**2 + ky**2)
    return k, kx, ky, kx_vec, ky_vec


def makeDim(N, d):
    if (N % 2) == 0:
        n = np.arange(-N/2, N/2)/N
    else:
        n = np.arange(-(N-1)/2, N/2)/N
    n[int(math.floor(N/2))] = 0
    return (2*np.pi/d) * n

if __name__ == "__main__":
    from scipy import io
    from show_img import show_three_images
    xb = io.loadmat("/root/data-fs/yueg_data/Data_skin_correct_distrib/Sig/val/sinograms/45model.mat")['test_sig1']
    yb = io.loadmat("/root/data-fs/yueg_data/Data_skin_correct_distrib/Initial/Img_trn_val/45model.mat")['initial']
    # img, x, z = pat_fft_linear_v3(xb[0:1250,:],1/62.5e6,0.106e-3,1500,0.106e-3*258,258)
    # show_three_images(xb, yb, img, 'recon_fft')
    # img_ubp, x, z = pat_ubp_linear(xb, 1/62.5e6, 0.106e-3, 1500, 0.106e-3*258, 258)
    # show_three_images(xb, yb, img_ubp, 'recon_ubp')
    f_fft,p0 = fft(xb)
    
    show_three_images(yb, p0.squeeze().detach().cpu().numpy(), f_fft.squeeze().detach().cpu().numpy(), 'recon_fft_paper2')
    show_three_images(yb, p0[:,:,0:256,:].squeeze().detach().cpu().numpy(), f_fft.squeeze().detach().cpu().numpy(), 'recon_fft_paper_cut')
    print('done')