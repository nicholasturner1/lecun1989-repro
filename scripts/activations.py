"""Collecting the outputs from each layer."""
import os
import argparse

import h5py
import torch
import numpy as np
from torch import nn

from lecun1989repro import models


def main(param_filename: str, output_dir: str, orig: bool = False) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # init model
    print("Loading model")
    model = models.Net() if orig else models.ModernNet()
    model.load_state_dict(torch.load(param_filename))
    model.eval()  # turning dropout off

    print("Loading data")
    Xtr, Ytr = torch.load("train1989.pt")
    Xte, Yte = torch.load("test1989.pt")

    print("Processing training set")
    acts_tr = collect_activations(model, Xtr)
    write_activations(output_dir, acts_tr, "train")

    print("Processing test set")
    acts_te = collect_activations(model, Xte)
    write_activations(output_dir, acts_te, "test")


def collect_activations(model: nn.Module, X: torch.tensor) -> dict[str, np.ndarray]:
    """Runs the network on each input and collects activations as np.ndarrays."""

    h1acts = np.zeros((X.size(0), 12, 8, 8), dtype=np.float32)
    h2acts = np.zeros((X.size(0), 12, 4, 4), dtype=np.float32)
    h3acts = np.zeros((X.size(0), 30), dtype=np.float32)
    outputs = np.zeros((X.size(0), 10), dtype=np.float32)

    for i in range(X.size(0)):
        o, h3, h2, h1 = model._forward(X[[i]])

        outputs[i, ...] = numpy(o)[:]
        h3acts[i, ...] = numpy(h3)[0][:]
        h2acts[i, ...] = numpy(h2)[0][:]
        h1acts[i, ...] = numpy(h1)[0][:]

    resultdict = {"h1": h1acts, "h2": h2acts, "h3": h3acts, "outputs": outputs}

    return resultdict


def numpy(t: torch.tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def write_activations(
    output_dir: str, activations: dict[str, np.ndarray], tag: str
) -> None:
    """Saves the activations to disk as an hdf5 file."""

    filename = os.path.join(output_dir, f"activations_{tag}.h5")
    with h5py.File(filename, "w") as f:
        for name, acts in activations.items():
            f.create_dataset(name, data=acts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("param_filename", type=str, help="Trained parameters.")
    ap.add_argument("output_dir", type=str, help="Output directory.")
    ap.add_argument("--orig", action="store_true")

    args = ap.parse_args()

    main(**vars(args))
