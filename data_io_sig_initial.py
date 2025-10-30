# -*- coding: utf-8 -*-
import numpy as np
from pathlib import Path
import re
import h5py
from scipy.io import loadmat    
#---------------------------------Read Me----------------------------------------------------
# 
# 将.mat数据保存为h5格式 
#---------------------------------------------------------------------------------------------

def _read_mat_any(path, key):
    try:
        with h5py.File(path, 'r') as f:
            return np.array(f[key])
    except OSError:
        m = loadmat(path)
        if key not in m:
            raise KeyError(f'{key} not found in {path}')
        return np.array(m[key])

def _sorted_files(d, pattern='*.mat', sort_numeric=True):
    paths = list(Path(d).glob(pattern))
    if sort_numeric:
        def _key(p):
            nums = re.findall(r'\d+', p.stem)
            return tuple(int(x) for x in nums) if nums else (p.stem,)
        paths.sort(key=_key)
    else:
        paths.sort()
    return paths

def load_sig_initial_dirs_2d(
    sig_train_dir, sig_val_dir, sig_test_dir,
    img_trn_val_dir, img_test_dir,
    data_key='test_sig1', img_key='initial',
    limit_train=None, limit_val=None, limit_test=None
):
    """
    读取并按“同名文件”在 Sig/*/sinogram 与 Initial/* 之间配对。
    输入 test_sig1: (Nt, Ny) 或 (Ny, Nt) -> 统一到 [N,1,Nt,Ny]
    金标准 initial: (Nx, Ny) -> 统一到 [N,1,Ny,Nx]
    返回对象 out.train / out.test，均带 next_batch() / selected_set() / ind
    """
    def _pair_and_stack(sig_dir, img_dir, limit):
        sig_files = _sorted_files(sig_dir)
        img_files = _sorted_files(img_dir)
        name2img = {p.stem: p for p in img_files}

        X_list, Y_list, names = [], [], []
        count = 0
        for p in sig_files:
            stem = p.stem
            if stem not in name2img:
                continue
            if limit is not None and count >= limit:
                break

            # 输入：test_sig1 (Nt, Ny) 或 (Ny, Nt)
            dt = np.squeeze(_read_mat_any(str(p), data_key))
            if dt.ndim != 2:
                raise ValueError(f'{p} -> {data_key} 必须是2D数组')
            # 统一到 (Nt, Ny)
            Nt, Ny = dt.shape
            # 若可能是 (Ny, Nt)，用更大的作为时间维不一定可靠；直接检查两种形状：
            # 假设 Ny 与金标准的 Ny 匹配，时间维一般更大，这里容错：如果 Nt < Ny 且 dt.T 更像 (Nt,Ny) 就转置
            if Nt < Ny:
                dt = dt.T
                Nt, Ny = dt.shape

            # 金标准：initial (Nx, Ny) -> (Ny, Nx)
            ip = name2img[stem]
            im = np.squeeze(_read_mat_any(str(ip), img_key))
            if im.ndim != 2:
                raise ValueError(f'{ip} -> {img_key} 必须是2D数组')
            Nx, Ny2 = im.shape
            if Ny2 != Ny:
                # 如不匹配，尝试转置
                if im.T.shape[0] == Ny:
                    im = im.T     # (Ny, Nx)
                    Ny2, Nx = im.shape
                else:
                    raise ValueError(f'{p} 与 {ip} 的 Ny 不一致: data Ny={Ny}, img Ny={Ny2}')

            # 堆叠到 batch 维，并加通道维
            X_list.append(dt[None, None, :, :])    # [1,1,Nt,Ny]
            Y_list.append(im.T[None, None, :, :])  # 注意：im 当前是 (Nx,Ny) 或 (Ny,Nx)
                                                   # 统一强制成 (Ny, Nx)：若上面未转置，这里再转置一次安全
            names.append(stem)
            count += 1

        if not X_list:
            raise RuntimeError(f'在 {sig_dir} 未找到可配对的 .mat')
        X = np.concatenate(X_list, axis=0)
        Y = np.concatenate(Y_list, axis=0)
        ind = np.arange(X.shape[0])
        return X, Y, ind, names

    # 合并 train + val（Img_trn_val 同时包含这两部分的金标准）
    X_trn, Y_trn, ind_trn, _ = _pair_and_stack(sig_train_dir, img_trn_val_dir, limit_train)
    X_val, Y_val, ind_val, _ = _pair_and_stack(sig_val_dir,   img_trn_val_dir, limit_val)
    X_tr = np.concatenate([X_trn, X_val], axis=0)
    Y_tr = np.concatenate([Y_trn, Y_val], axis=0)
    ind_tr_all = np.arange(X_tr.shape[0])

    X_te, Y_te, ind_te, _ = _pair_and_stack(sig_test_dir, img_test_dir, limit_test)

    class _DataSet:
        def __init__(self, data, true, indices):
            self._data = data; self._true = true
            self._data_orig = data; self._true_orig = true
            self._num_examples = data.shape[0]
            self._index_in_epoch = 0; self._epochs_completed = 0
            self._ind = indices
        @property
        def ind(self): return self._ind
        def next_batch(self, b):
            s, e = self._index_in_epoch, self._index_in_epoch + b
            if e > self._num_examples:
                perm = np.random.permutation(self._num_examples)
                self._data, self._true = self._data[perm], self._true[perm]
                s, e = 0, b; self._epochs_completed += 1
            self._index_in_epoch = e
            return self._data[s:e], self._true[s:e]
        def selected_set(self, idx):
            return self._data_orig[idx], self._true_orig[idx]

    class Bundle: pass
    out = Bundle()
    out.train = _DataSet(X_tr, Y_tr, ind_tr_all)
    out.test  = _DataSet(X_te, Y_te, ind_te)
    return out
#---------------------------------------------------------------------------------------------
#  将.mat数据保存为h5格式
#---------------------------------------------------------------------------------------------
def inspect_h5_file(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            # 输出文件的所有数据集名称
            print("文件中的数据集名称：")
            for key in f.keys():
                print(f"数据集名: {key}")
                # 输出每个数据集的尺寸
                dataset = f[key]
                print(f"数据集形状: {dataset.shape}")
                print(f"数据集类型: {dataset.dtype}")
                print('-' * 40)
    except Exception as e:
        print(f"出现错误: {e}")


def _read_mat_any(file_path, key):
    # 读取MAT文件内容的代码（需要根据具体库和格式来实现）
    # 例如：使用 scipy.io 或 h5py 读取mat文件，获取指定键的数据。
    from scipy.io import loadmat
    mat_data = loadmat(file_path)
    return mat_data[key]

def _sorted_files(directory):
    # 返回文件夹下所有文件的排序列表（假设文件是以文件名排序的）
    from pathlib import Path
    return sorted(Path(directory).glob("*.mat"))

def save_dataset_as_h5(sig_train_dir, sig_val_dir, sig_test_dir, img_trn_val_dir, img_test_dir, data_key='test_sig1', img_key='initial', output_file='data_cache.h5', chunk_size=100):
    """
    将数据保存为 HDF5 格式
    """
    try:
        # 使用 'w' 模式，完全清空旧数据并创建新文件
        with h5py.File(output_file, 'w') as f:
            def _pair_and_stack(sig_dir, img_dir):
                sig_files = _sorted_files(sig_dir)
                img_files = _sorted_files(img_dir)
                name2img = {p.stem: p for p in img_files}

                
                X_list, Y_list, names = [], [], []
                for p in sig_files:
                    stem = p.stem
                    if stem not in name2img:
                        continue

                    # 输入数据 (Nt, Ny) 或 (Ny, Nt)
                    dt = np.squeeze(_read_mat_any(str(p), data_key))
                    if dt.ndim != 2:
                        raise ValueError(f'{p} -> {data_key} 必须是2D数组')
                    Nt, Ny = dt.shape
                    if Nt < Ny:
                        dt = dt.T
                        Nt, Ny = dt.shape

                    # 金标准数据 (Nx, Ny) 转换为 (Ny, Nx)
                    ip = name2img[stem]
                    im = np.squeeze(_read_mat_any(str(ip), img_key))
                    if im.ndim != 2:
                        raise ValueError(f'{ip} -> {img_key} 必须是2D数组')

                    Nx, Ny2 = im.shape
                    if Ny2 != Ny:
                        if im.T.shape[0] == Ny:
                            im = im.T
                            Ny2, Nx = im.shape
                        else:
                            raise ValueError(f'{p} 与 {ip} 的 Ny 不一致: data Ny={Ny}, img Ny={Ny2}')

                    X_list.append(dt[None, None, :, :])  # [1, 1, Nt, Ny]
                    Y_list.append(im[None, None, :, :])  # [1, 1, Ny, Nx]
                    names.append(stem)

                X = np.concatenate(X_list, axis=0)
                Y = np.concatenate(Y_list, axis=0)
                return X, Y, names

            # 处理并保存各个数据集
            X_trn, Y_trn, names_trn = _pair_and_stack(sig_train_dir, img_trn_val_dir)
            X_val, Y_val, names_val = _pair_and_stack(sig_val_dir, img_trn_val_dir)
            X_te, Y_te, names_te = _pair_and_stack(sig_test_dir, img_test_dir)

            print("开始保存数据到 HDF5 文件 (分块保存)...")
                
            def write_in_chunks(dataset_name, data, chunk_size):
                """将数据分块写入 HDF5 文件"""
                total_size = data.shape[0]
                f.create_dataset(dataset_name, data=data[:chunk_size], maxshape=(None,)+data.shape[1:],chunks=(chunk_size,)+data.shape[1:],dtype=data.dtype)
                # 循环写入数据块
                for start in range(chunk_size, total_size, chunk_size):
                    end = min(start + chunk_size, total_size)
                    print(f"写入 {dataset_name} 数据块 [{start}, {end})...")
                    if end > f[dataset_name].shape[0]:
                        f[dataset_name].resize((end,) + f[dataset_name].shape[1:])
                    f[dataset_name][start:end] = data[start:end]  # 写入数据块


            # 写入训练数据（分块写入）
            write_in_chunks('X_trn', X_trn, chunk_size)
            write_in_chunks('Y_trn', Y_trn, chunk_size)
            write_in_chunks('X_val', X_val, chunk_size)
            write_in_chunks('Y_val', Y_val, chunk_size)
            write_in_chunks('X_tst', X_te, chunk_size)
            write_in_chunks('Y_tst', Y_te, chunk_size)
            
            # 保存文件名列表
            dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset('names_trn', data=names_trn, dtype=dt)
            f.create_dataset('names_val', data=names_val, dtype=dt)
            f.create_dataset('names_tst', data=names_te, dtype=dt)
            print("数据集已成功分块保存为 HDF5 格式")
            
    except Exception as e:
        print(f"出现错误: {e}")




if __name__ == "__main__":
    
    root = '/root/data-fs/yueg_data/Data_skull_correct_distrib'
    save_dataset_as_h5(
        sig_train_dir   = f'{root}/Sig/train/sinograms',
        sig_val_dir     = f'{root}/Sig/val/sinograms',
        sig_test_dir    = f'{root}/Sig/test/sinograms',
        img_trn_val_dir = f'{root}/Initial/Img_trn_val',
        img_test_dir    = f'{root}/Initial/Img_test',
        output_file='/root/data-fs/yueg_data/Data_skull_h5/data_cache.h5',
        chunk_size=100
    )
    inspect_h5_file("/root/data-fs/yueg_data/Data_skull_h5/data_cache.h5")
    
    
               
    # root = '/root/data-fs/yueg_data/Data_skin_correct_distrib'
    # save_dataset_as_h5(
    #     sig_train_dir   = f'{root}/Sig/train/sinograms',
    #     sig_val_dir     = f'{root}/Sig/val/sinograms',
    #     sig_test_dir    = f'{root}/Sig/test/sinograms',
    #     img_trn_val_dir = f'{root}/Initial/Img_trn_val',
    #     img_test_dir    = f'{root}/Initial/Img_test',
    #     output_file='/root/data-fs/yueg_data/Data_skin_h5/data_cache.h5',
    #     chunk_size=100
    # )
    # inspect_h5_file("/root/data-fs/yueg_data/Data_skin_h5/data_cache.h5")
