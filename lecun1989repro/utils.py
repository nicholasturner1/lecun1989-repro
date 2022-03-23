"""Miscellaneous utilties."""
import h5py
import numpy as np


def read_h5_dict(filename: str) -> dict[str, np.ndarray]:
    """Reads the first-level attributes from an hdf5 file."""
    contents = dict()

    with h5py.File(filename) as f:
        for k in f.keys():
            contents[k] = f[k][()]

    return contents
