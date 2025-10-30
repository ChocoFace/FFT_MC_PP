from data_io_sig_initial import load_sig_initial_dirs_2d
from kgrids_pytorch_kspace_2D import kspaceKGrids2D
from fft_mc_pp_2D import train_fft_mc_pp_2d, eval_fft_mc_pp_2d
import numpy as np
import h5py



# 打开 HDF5 文件（'r' 表示只读模式）
with h5py.File('/root/data-fs/yueg_data/Data_skin_h5/data_cache.h5', 'r') as f:
    # 打印文件中的所有顶级组（Groups）和数据集（Datasets）
    print("文件结构：")
    def print_structure(name, obj):
        print(name)
        # 如果是数据集，打印其形状、数据类型
        if isinstance(obj, h5py.Dataset):
            print(f"  类型：数据集，形状：{obj.shape}，数据类型：{obj.dtype}")
        # 如果是组，打印其属性
        elif isinstance(obj, h5py.Group):
            print(f"  类型：组，属性：{dict(obj.attrs)}")
    f.visititems(print_structure)
    
dataset_file='/root/data-fs/yueg_data/Data_skin_h5/data_cache.h5'
with h5py.File(dataset_file, 'r') as f:
    pTy = np.array(f['X_tst'][0, 0, :, :], dtype=np.float32) 
# 物理参数（与你原始脚本一致）
# dx=5.3e-5*2; dy=5.3e-5*2; dt=3*5/3*1e-8; c=1500; NtFactor=5
dx=0.106e-3; dy=0.106e-3; dt=1/62.5e6; c=1500; NtFactor=5


# 用一条测试样本构建 k-space 插值网格（p(t,y)）
kg = kspaceKGrids2D(dx,dy,dt,c,NtFactor)
sf, ky, w, kyI, wI = kg.inverse(pTy.astype(np.float32))
print('sf shape:',  sf.shape)   # 通常 ~ (Nt', Ny')  (插值目标分辨率)
print('ky shape:',  ky.shape)   # (Ny,)
print('w shape:',   w.shape)    # (Nt,)
print('kyI shape:', kyI.shape)  # (Nt', Ny') 目标网格
print('wI shape:',  wI.shape)   # (Nt', Ny') 目标网格 (频率半径)

# 训练 FFT + MC + PP
expName = 'fft_mc_pp_2d_sig_initial'
filePath = './Results/'
_ = train_fft_mc_pp_2d(ds, sf, ky, w, kyI, wI, c, NtFactor,
                       expName, filePath,
                       bSize=1, trainIter=50000, lr=1e-2, useTensorboard=True)

# 评估并保存 p0 / p0MC / p0MCPP
eval_fft_mc_pp_2d(ds, expName, filePath, saveResults=True)
