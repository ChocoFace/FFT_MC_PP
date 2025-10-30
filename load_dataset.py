# pip install h5py torch torchvision
import h5py, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

def pad_to_multiple_hw(arr_chw, multiple=32, value=0.0):
    # arr_chw: (C,H,W) numpy
    C,H,W = arr_chw.shape
    pad_h = (multiple - (H % multiple)) % multiple
    pad_w = (multiple - (W % multiple)) % multiple
    if pad_h or pad_w:
        arr_chw = np.pad(arr_chw, ((0,0),(0,pad_h),(0,pad_w)), mode="constant", constant_values=value)
    return arr_chw

class H5PairedDataset(Dataset):
    """
    以 X_trn / Y_trn, X_val / Y_val, X_tst / Y_tst 这类命名读取；
    懒打开 HDF5，避免在多个 worker 之间共享同一个文件句柄。
    """
    def __init__(self, h5_path, split="trn", pad_multiple=32, norm="none"):
        self.h5_path = h5_path
        self.split = split
        self.pad_multiple = pad_multiple
        self.norm = norm  # "none" | "minmax" | "standard"
        self._file = None
        self._X = None
        self._Y = None
        self.x_key = f"X_{split}"
        self.y_key = f"Y_{split}"

    def _ensure_open(self):
        if self._file is None:
            # SWMR 只读，适合多进程并发读取
            self._file = h5py.File(self.h5_path, "r", libver="latest", swmr=True)
            self._X = self._file[self.x_key]  # (N,1,H,W)
            self._Y = self._file[self.y_key]  # (N,1,h,w)

    def __len__(self):
        self._ensure_open()
        return self._X.shape[0]

    def _normalize(self, x):

        return x

    def __getitem__(self, idx):
        self._ensure_open()
        # 逐样本懒加载 + dtype 转 float32
        x = self._X[idx].astype(np.float32)  # (1,H,W)
        y = self._Y[idx].astype(np.float32)  # (1,h,w)

        # 转 torch.Tensor
        x = torch.from_numpy(x)  # (1,H',W')
        y = torch.from_numpy(y)  # (1,256,256)
        return x, y,idx

def make_loader(h5_path, split, batch_size=4, shuffle=True, workers=4):
    ds = H5PairedDataset(h5_path, split=split, pad_multiple=32, norm="none")
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, pin_memory=True, persistent_workers=(workers>0)
    )

if __name__ == "__main__":
    # 简单测试
    h5_path = "/root/data-fs/yueg_data/Data_skin_h5/data_cache.h5"  # 替换为你的 HDF5 文件路径
    # ds = H5PairedDataset(h5_path, split="trn", pad_multiple=32, norm="none")
    # print(f"数据集大小: {len(ds)}")
    # x,y,idx = ds[0]
    # print(f"x shape: {x.shape}, y shape: {y.shape}")
    # DataLoader 测试
    loader = make_loader(h5_path, split="trn", batch_size=2, shuffle=False, workers=2)
    for xb,yb,idx in loader:
        print(f"batch x shape: {xb.shape}, y shape: {yb.shape}")
        break
    # === 用法 ===
    train_loader = make_loader("your_dataset.h5", "trn", batch_size=4, shuffle=True,  workers=4)
    val_loader   = make_loader("your_dataset.h5", "val", batch_size=4, shuffle=False, workers=2)

    # ====== 一个最小化训练循环骨架（以 L1 为例；你可换成自己的模型和损失）======
    import torch.nn as nn

    class DummyModel(nn.Module):
        # 把 (1,2048,256) -> (1,256,256) 的占位模型，换成你的 UNet/ResNet 等
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((256,256)),
            )
            self.out = nn.Conv2d(32, 1, 1)
        def forward(self, x):
            x = self.enc(x)
            x = self.out(x)
            return x

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DummyModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    def center_crop_to(t, hw):
        h, w = hw
        _,_,H,W = t.shape
        top = max((H - h)//2, 0); left = max((W - w)//2, 0)
        return t[:,:, top:top+h, left:left+w]

    for epoch in range(5):
        model.train()
        for x,y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                pred = model(x)
                # 若输出尺寸与标签不一致，做中心裁剪到标签大小
                if pred.shape[-2:] != y.shape[-2:]:
                    pred = center_crop_to(pred, y.shape[-2:])
                loss = crit(pred, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        # 简单校验
        model.eval()
        with torch.no_grad():
            x,y = next(iter(val_loader))
            x,y = x.to(device), y.to(device)
            p = model(x)
            if p.shape[-2:] != y.shape[-2:]:
                p = center_crop_to(p, y.shape[-2:])
            val_loss = crit(p,y).item()
        print(f"epoch {epoch} val L1: {val_loss:.4f}")

