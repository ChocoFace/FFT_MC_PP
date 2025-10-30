# ===============================
# FILE: data_io_2D.py  (MAT loader -> torch-ready)
# ===============================
import numpy as np
import h5py
from scipy.io import loadmat

# Expected dataset names (configurable):
IMG_KEY = 'imagesTrue'   # shape (Ny, Nx, N)
DATA_KEY = 'dataTrue'    # shape (Ny, Nt, N)


def _read_mat_any(path, key):
    """Reads either MATLAB v7.3 (HDF5) via h5py or older via scipy.io.loadmat.
    Returns a numpy array.
    """
    try:
        with h5py.File(path, 'r') as f:
            d = f[key]
            arr = np.array(d)
            # h5py gives Fortran order sometimes
            return np.array(arr)
    except OSError:
        # Not HDF5 -> try old MAT
        m = loadmat(path)
        if key not in m:
            raise KeyError(f'{key} not found in {path}')
        return np.array(m[key])


def load_dataset_2d(train_mat, test_mat, train_indices, test_indices,
                    img_key=IMG_KEY, data_key=DATA_KEY):
    class DataSet:
        def __init__(self, data, true, indices):
            self._data = data  # [N,1,Nt,Ny]
            self._true = true  # [N,1,Ny,Nx]
            self._data_orig = data
            self._true_orig = true
            self._num_examples = data.shape[0]
            self._epochs_completed = 0
            self._index_in_epoch = 0
            self._ind = indices
        @property
        def ind(self): return self._ind
        def next_batch(self, b):
            s = self._index_in_epoch; e = s + b
            if e > self._num_examples:
                perm = np.random.permutation(self._num_examples)
                self._data = self._data[perm]
                self._true = self._true[perm]
                s = 0; e = b; self._epochs_completed += 1
            self._index_in_epoch = e
            return self._data[s:e], self._true[s:e]
        def selected_set(self, idx):
            return self._data_orig[idx], self._true_orig[idx]
    class Bundle: pass

    def _prep(path, idxs):
        imgs = _read_mat_any(path, img_key)      # (Ny, Nx, N)
        data = _read_mat_any(path, data_key)     # (Ny, Nt, N)
        # move sample dim to front and add channel dims
        imgs = np.moveaxis(imgs, -1, 0)[:, None, :, :]      # [N,1,Ny,Nx]
        data = np.moveaxis(data, -1, 0)[:, None, :, :]      # [N,1,Ny,Nt]
        # but we want [N,1,Nt,Ny]
        data = np.swapaxes(data, 2, 3)
        return data[idxs], imgs[idxs]

    tr_data, tr_imgs = _prep(train_mat, train_indices)
    te_data, te_imgs = _prep(test_mat, test_indices)

    out = Bundle()
    out.train = DataSet(tr_data, tr_imgs, train_indices)
    out.test = DataSet(te_data, te_imgs, test_indices)
    return out
