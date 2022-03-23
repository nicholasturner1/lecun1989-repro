"""
Running this script eventually gives:
23
eval: split train. loss 4.073383e-03. error 0.62%. misses: 45
eval: split test . loss 2.838382e-02. error 4.09%. misses: 82
"""

import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter  # pip install tensorboardX

from lecun1989repro import models


def main(learning_rate: float, output_dir: str) -> None:
    # init rng
    torch.manual_seed(1337)
    np.random.seed(1337)
    torch.use_deterministic_algorithms(True)

    # set up logging
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    writer = SummaryWriter(args.output_dir)

    # init a model
    model = models.Net()
    print("model stats:")
    print(
        "# params:      ", sum(p.numel() for p in model.parameters())
    )  # in paper total is 9,760
    print("# MACs:        ", model.macs)
    print("# activations: ", model.acts)

    # init data
    Xtr, Ytr = torch.load("train1989.pt")
    Xte, Yte = torch.load("test1989.pt")

    # init optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    def eval_split(split):
        # eval the full train/test set, batched implementation for efficiency
        model.eval()
        X, Y = (Xtr, Ytr) if split == "train" else (Xte, Yte)
        Yhat = model(X)
        loss = torch.mean((Y - Yhat) ** 2)
        err = torch.mean((Y.argmax(dim=1) != Yhat.argmax(dim=1)).float())
        print(
            f"eval: split {split:5s}. loss {loss.item():e}. error {err.item()*100:.2f}%. misses: {int(err.item()*Y.size(0))}"
        )
        writer.add_scalar(f"error/{split}", err.item() * 100, pass_num)
        writer.add_scalar(f"loss/{split}", loss.item(), pass_num)

    # train
    for pass_num in range(23):

        # perform one epoch of training
        model.train()
        for step_num in range(Xtr.size(0)):

            # fetch a single example into a batch of 1
            x, y = Xtr[[step_num]], Ytr[[step_num]]

            # forward the model and the loss
            yhat = model(x)
            loss = torch.mean((y - yhat) ** 2)

            # calculate the gradient and update the parameters
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # after epoch epoch evaluate the train and test error / metrics
        print(pass_num + 1)
        eval_split("train")
        eval_split("test")

    # save final model to file
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a 1989 LeCun ConvNet on digits")

    parser.add_argument(
        "--learning-rate", "-l", type=float, default=0.03, help="SGD learning rate"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="out/base",
        help="output directory for training logs",
    )

    args = parser.parse_args()

    print(vars(args))
    main(**vars(args))
