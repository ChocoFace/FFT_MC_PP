import h5py
import numpy as np

class H5Dataset:
    def __init__(self, h5_path, x_key='X_trn', y_key='Y_trn', is_train=True):
        """
        初始化H5数据集加载器
        :param h5_path: H5文件路径
        :param x_key: H5中输入数据的键（如'X_trn'）
        :param y_key: H5中标签数据的键（如'Y_trn'）
        :param is_train: 是否为训练集（训练集需要随机打乱，测试集通常按顺序取）
        """
        self.h5_path = h5_path
        self.x_key = x_key
        self.y_key = y_key
        self.is_train = is_train  # 标记是否为训练集（控制是否打乱）
        self.bSize = 1  # 固定batch size为1（根据原代码要求）
        
        # 打开H5文件并加载数据集
        self.f = h5py.File(self.h5_path, 'r')
        self.Xds = self.f[self.x_key]  # 输入数据（shape [N,1,Nt,Ny]）
        self.Yds = self.f[self.y_key]  # 标签数据（shape [N,1,Ny,Nx]）
        self.N = self.Xds.shape[0]  # 样本总数
        
        # 初始化指针和打乱顺序（训练集需要）
        self.ptr = 0
        self.order = np.arange(self.N)  # 初始顺序为0~N-1
        if self.is_train:
            self.order = np.random.permutation(self.N)  # 训练集随机打乱

    def next_batch(self, B=None):
        """
        获取下一个batch的数据
        :param B: batch size（默认为类初始化时的bSize=1）
        :return: x_batch, y_batch（输入和标签的numpy数组）
        """
        B = self.bSize if B is None else B
        if B != 1:
            print('[WARN] 当前实现要求 bSize==1（k-space 插值使用单样本）。已强制使用 batch=1。')
            B = 1
        
        # 若指针超出范围，重新打乱（仅训练集）并重置指针
        if self.ptr + B > self.N:
            if self.is_train:
                self.order = np.random.permutation(self.N)  # 训练集重新打乱
            else:
                self.order = np.arange(self.N)  # 测试集按顺序循环
            self.ptr = 0
        
        # 获取当前batch的索引
        idx = self.order[self.ptr: self.ptr + B]
        self.ptr += B
        
        # 读取数据（转换为numpy数组）
        x_batch = np.asarray(self.Xds[idx])  # shape (B,1,Nt,Ny)
        y_batch = np.asarray(self.Yds[idx])  # shape (B,1,Ny,Nx)
        
        return x_batch, y_batch

    def close(self):
        """关闭H5文件（训练结束后调用）"""
        self.f.close()


# 数据集管理器（用于统一管理训练集、测试集，类似你需要的`dataset.train`调用方式）
class DatasetManager:
    def __init__(self, train_dataset, val_dataset, test_dataset=None):
        self.train = train_dataset  # 训练集加载器
        self.val = val_dataset      # 验证集加载器
        self.test = test_dataset    # 测试集加载器（可选）
        
def dataset(h5_path, x_key='X_trn', y_key='Y_trn', val_x_key='X_val', val_y_key='Y_val', test_x_key='X_tst', test_y_key='Y_tst'):
    """
    加载H5数据集并返回DatasetManager
    :param h5_path: H5文件路径
    :param x_key: 训练集输入数据键
    :param y_key: 训练集标签数据键
    :param val_x_key: 验证集输入数据键
    :param val_y_key: 验证集标签数据键
    :param test_x_key: 测试集输入数据键
    :param test_y_key: 测试集标签数据键
    :return: DatasetManager实例
    """
    train_dataset = H5Dataset(h5_path, x_key, y_key, is_train=True)
    val_dataset = H5Dataset(h5_path, val_x_key, val_y_key, is_train=False)
    # 测试集可选加载
    test_dataset = H5Dataset(h5_path, test_x_key, test_y_key, is_train=False)
    return DatasetManager(train_dataset, val_dataset, test_dataset)