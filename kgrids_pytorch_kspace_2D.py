# ===============================
# FILE: kgrids_pytorch_kspace_2D.py
# ===============================
import numpy as np
import math

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


