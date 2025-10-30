# ===============================
# FILE: run_2d_template.py (usage sketch)
# ===============================
"""
from kgrids_pytorch_kspace_2D import kspaceKGrids2D
from nets_pytorch_kspace_2D import train_postprocess_2d
import h5py, numpy as np

# Data I/O (you will adapt your HDF5 to 2D):
# imagesTrue(y,x,sample) -> reshape to [N,1,Ny,Nx]
# dataTrue(sensor_y,time,sample) -> reshape to [N,1,Nt,Ny]

# Example params (keep identical semantics):
dx=5.3e-5*2; dy=5.3e-5*2; dt=3*5/3*1e-8; c=1500; NtFactor=5
kg = kspaceKGrids2D(dx,dy,dt,c,NtFactor)
# Build grids from one sample of p(t,y)
# sf, ky, w, kyI, wI = kg.inverse(pTy)
# model = train_postprocess_2d(dataSet, sf, ky, w, kyI, wI, c, NtFactor, expName, filePath)
"""