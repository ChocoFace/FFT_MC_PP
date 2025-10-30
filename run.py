from data_io_sig_initial import load_sig_initial_dirs_2d
from kgrids_pytorch_kspace_2D import kspaceKGrids2D
from fft_mc_pp_2D import train_fft_mc_pp_2d, eval_fft_mc_pp_2d
import numpy as np
from dataset import dataset

ds = dataset(
    h5_path="/root/data-fs/yueg_data/Data_skin_h5/data_cache.h5",
    x_key='X_trn',
    y_key='Y_trn',
    val_x_key='X_val',
    val_y_key='Y_val',
    test_x_key='X_tst',
    test_y_key='Y_tst',
)
# 物理参数（与你原始脚本一致）
# dx=5.3e-5*2; dy=5.3e-5*2; dt=3*5/3*1e-8; c=1500; NtFactor=5
dx=0.106e-3; dy=0.106e-3; dt=1/62.5e6; c=1500; NtFactor=5


# 用一条测试样本构建 k-space 插值网格（p(t,y)）
xb,yb = ds.train.next_batch(B=1)  # [Nt, Ny]
kg = kspaceKGrids2D(dx,dy,dt,c,NtFactor)
sf, ky, w, kyI, wI = kg.inverse(xb.squeeze().astype(np.float32))

# 训练 FFT + MC + PP
expName = 'fft_mc_pp_2d_sig_initial'
filePath = './Results/'
_ = train_fft_mc_pp_2d(ds, sf, ky, w, kyI, wI, c, NtFactor,
                       expName, filePath,
                       bSize=1, trainIter=50000, lr=1e-2, useTensorboard=True)

# 评估并保存 p0 / p0MC / p0MCPP
eval_fft_mc_pp_2d(ds, expName, filePath, saveResults=True)
