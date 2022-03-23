"""

repro.py gives:
23
eval: split train. loss 4.073383e-03. error 0.62%. misses: 45
eval: split test . loss 2.838382e-02. error 4.09%. misses: 82

we can try to use our knowledge from 33 years later to improve on this,
but keeping the model size same.

Change 1: replace tanh on last layer with FC and use softmax. Had to
lower the learning rate to 0.01 as well. This improves the optimization
quite a lot, we now crush the training set:
23
eval: split train. loss 9.536698e-06. error 0.00%. misses: 0
eval: split test . loss 9.536698e-06. error 4.38%. misses: 87

Change 2: change from SGD to AdamW with LR 3e-4 because I find this
to be significantly more stable and requires little to no tuning. Also
double epochs to 46. I decay the LR to 1e-4 over course of training.
These changes make it so optimization is not culprit of bad performance
with high probability. We also seem to improve test set a bit:
46
eval: split train. loss 0.000000e+00. error 0.00%. misses: 0
eval: split test . loss 0.000000e+00. error 3.59%. misses: 72

Change 3: since we are overfitting we can introduce data augmentation,
e.g. let's intro a shift by at most 1 pixel in both x/y directions. Also
because we are augmenting we again want to bump up training time, e.g.
to 60 epochs:
60
eval: split train. loss 8.780676e-04. error 1.70%. misses: 123
eval: split test . loss 8.780676e-04. error 2.19%. misses: 43

Change 4: we want to add dropout at the layer with most parameters (H3),
but in addition we also have to shift the activation function to relu so
that dropout makes sense. We also bring up iterations to 80:
80
eval: split train. loss 2.601336e-03. error 1.47%. misses: 106
eval: split test . loss 2.601336e-03. error 1.59%. misses: 32

To be continued...
"""

import os
import json
import argparse

import numpy as np
import torch
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
    model = models.ModernNet()
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
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    def eval_split(split):
        # eval the full train/test set, batched implementation for efficiency
        model.eval()
        X, Y = (Xtr, Ytr) if split == "train" else (Xte, Yte)
        Yhat = model(X)
        loss = F.cross_entropy(yhat, y.argmax(dim=1))
        err = torch.mean((Y.argmax(dim=1) != Yhat.argmax(dim=1)).float())
        print(
            f"eval: split {split:5s}."
            f" loss {loss.item():e}."
            f" error {err.item()*100:.2f}%."
            f" misses: {int(err.item()*Y.size(0))}"
        )
        writer.add_scalar(f"error/{split}", err.item() * 100, pass_num)
        writer.add_scalar(f"loss/{split}", loss.item(), pass_num)

    # train
    for pass_num in range(80):

        # learning rate decay
        alpha = pass_num / 79
        for g in optimizer.param_groups:
            g["lr"] = (1 - alpha) * args.learning_rate + alpha * (
                args.learning_rate / 3
            )

        # perform one epoch of training
        model.train()
        for step_num in range(Xtr.size(0)):

            # fetch a single example into a batch of 1
            x, y = Xtr[[step_num]], Ytr[[step_num]]

            # forward the model and the loss
            yhat = model(x)
            loss = F.cross_entropy(yhat, y.argmax(dim=1))

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
    parser = argparse.ArgumentParser(
        description="Train a 2022 but mini ConvNet on digits"
    )

    parser.add_argument(
        "--learning-rate", "-l", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="out/modern",
        help="output directory for training logs",
    )

    args = parser.parse_args()

    print(vars(args))
    main(**vars(args))
