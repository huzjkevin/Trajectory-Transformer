import argparse
import baselineUtils
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from transformer.batch import subsequent_mask
from torch.optim import Adam, SGD, RMSprop, Adagrad
from transformer.noam_opt import NoamOpt
import individual_TF
import numpy as np
import scipy.io
import json
import pickle
import datetime
import logging
import random
import yaml

from torch.utils.tensorboard import SummaryWriter


def main():
    parser = argparse.ArgumentParser(
        description="Train the individual Transformer model"
    )
    parser.add_argument("--dataset_folder", type=str, default="datasets")
    parser.add_argument("--dataset_name", type=str, default="trajectory_combined")
    parser.add_argument("--obs", type=int, default=8)
    parser.add_argument("--preds", type=int, default=12)
    parser.add_argument("--emb_size", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--val_size", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--validation_epoch_start", type=int, default=30)
    parser.add_argument("--resume_train", action="store_true")
    parser.add_argument("--delim", type=str, default="\t")
    parser.add_argument("--name", type=str, default="trajectory_combined")
    parser.add_argument("--factor", type=float, default=1.0)
    parser.add_argument("--save_step", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--evaluate", type=bool, default=True)
    parser.add_argument("--model_pth", type=str)

    args = parser.parse_args()
    model_name = args.name

    curr_time = datetime.datetime.now()
    output_dir = f"exp_{args.dataset_name}_{curr_time.strftime('%Y%m%d%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_dir = os.path.join(output_dir, f"IndividualTF_ckpts")
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_output_dir = os.path.join(output_dir, f"IndividualTF_outputs")
    os.makedirs(model_output_dir, exist_ok=True)

    # keep track of console outputs and experiment settings
    baselineUtils.set_logger(os.path.join(output_dir, f"train_{args.dataset_name}.log"))
    config_file = open(
        os.path.join(output_dir, f"config_{args.dataset_name}.yaml"), "w"
    )
    yaml.dump(args, config_file)
    tensorboard_dir = os.path.join(output_dir, "tensorboard")
    log = SummaryWriter(tensorboard_dir)

    seed = 72
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    log.add_scalar("eval/mad", 0, 0)
    log.add_scalar("eval/fad", 0, 0)
    device = torch.device("cuda")

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")

    args.verbose = True

    ## creation of the dataloaders for train and validation
    if args.val_size == 0:
        train_dataset, _ = baselineUtils.create_dataset(
            args.dataset_folder,
            args.dataset_name,
            0,
            args.obs,
            args.preds,
            delim=args.delim,
            train=True,
            verbose=args.verbose,
        )
        val_dataset, _ = baselineUtils.create_dataset(
            args.dataset_folder,
            args.dataset_name,
            0,
            args.obs,
            args.preds,
            delim=args.delim,
            train=False,
            verbose=args.verbose,
        )
    else:
        train_dataset, val_dataset = baselineUtils.create_dataset(
            args.dataset_folder,
            args.dataset_name,
            args.val_size,
            args.obs,
            args.preds,
            delim=args.delim,
            train=True,
            verbose=args.verbose,
        )

    test_dataset, _ = baselineUtils.create_dataset(
        args.dataset_folder,
        args.dataset_name,
        0,
        args.obs,
        args.preds,
        delim=args.delim,
        train=False,
        eval=True,
        verbose=args.verbose,
    )

    model = individual_TF.IndividualTF(
        2,
        3,
        3,
        N=args.layers,
        d_model=args.emb_size,
        d_ff=2048,
        h=args.heads,
        dropout=args.dropout,
        mean=[0, 0],
        std=[0, 0],
    ).to(device)
    if args.resume_train:
        model.load_state_dict(torch.load(f"{checkpoint_dir}/{args.model_pth}"))

    tr_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        collate_fn=baselineUtils.collate_fn,
    )
    val_dl = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        collate_fn=baselineUtils.collate_fn,
    )
    test_dl = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
        collate_fn=baselineUtils.collate_fn,
    )

    # optim = SGD(list(a.parameters())+list(model.parameters())+list(generator.parameters()),lr=0.01)
    # sched=torch.optim.lr_scheduler.StepLR(optim,0.0005)
    optim = NoamOpt(
        args.emb_size,
        args.factor,
        len(tr_dl) * args.warmup,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
    )
    # optim=Adagrad(list(a.parameters())+list(model.parameters())+list(generator.parameters()),lr=0.01,lr_decay=0.001)
    epoch = 0

    # mean=train_dataset[:]['src'][:,1:,2:4].mean((0,1))
    mean = np.concatenate(
        (train_dataset.data["src"][:, 1:, 2:4], train_dataset.data["trg"][:, :, 2:4]), 1
    ).mean((0, 1))
    # std=train_dataset[:]['src'][:,1:,2:4].std((0,1))
    std = np.concatenate(
        (train_dataset.data["src"][:, 1:, 2:4], train_dataset.data["trg"][:, :, 2:4]), 1
    ).std((0, 1))
    mean = torch.from_numpy(mean)
    std = torch.from_numpy(std)

    means = []
    stds = []
    for i in np.unique(train_dataset.data["dataset"]):
        ind = train_dataset.data["dataset"] == i
        means.append(
            torch.from_numpy(
                np.concatenate(
                    (
                        train_dataset.data["src"][ind, 1:, 2:4],
                        train_dataset.data["trg"][ind, :, 2:4],
                    ),
                    1,
                ).mean((0, 1))
            )
        )
        stds.append(
            torch.from_numpy(
                np.concatenate(
                    (
                        train_dataset.data["src"][ind, 1:, 2:4],
                        train_dataset.data["trg"][ind, :, 2:4],
                    ),
                    1,
                ).std((0, 1))
            )
        )

    mean = torch.stack(means).mean(0)
    std = torch.stack(stds).mean(0)

    scipy.io.savemat(
        f"{output_dir}/norm.mat",
        {"mean": mean.cpu().numpy(), "std": std.cpu().numpy()},
    )

    while epoch < args.max_epoch:
        epoch_loss = 0
        model.train()

        for id_b, batch in enumerate(tr_dl):

            optim.optimizer.zero_grad()
            seq_start_end = batch["seq_start_end"]
            inp = (batch["src"][:, 1:, 2:4].to(device) - mean.to(device)) / std.to(
                device
            )
            target = (batch["trg"][:, :-1, 2:4].to(device) - mean.to(device)) / std.to(
                device
            )

            target_c = torch.zeros((target.shape[0], target.shape[1], 1)).to(device)
            target = torch.cat((target, target_c), -1)
            start_of_seq = (
                torch.Tensor([0, 0, 1])
                .unsqueeze(0)
                .unsqueeze(1)
                .repeat(target.shape[0], 1, 1)
                .to(device)
            )

            dec_inp = torch.cat((start_of_seq, target), 1)

            src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(device)
            trg_att = (
                subsequent_mask(dec_inp.shape[1])
                .repeat(dec_inp.shape[0], 1, 1)
                .to(device)
            )

            pred = model(inp, dec_inp, src_att, trg_att, seq_start_end)

            loss = (
                F.pairwise_distance(
                    pred[:, :, 0:2].contiguous().view(-1, 2),
                    (
                        (batch["trg"][:, :, 2:4].to(device) - mean.to(device))
                        / std.to(device)
                    )
                    .contiguous()
                    .view(-1, 2)
                    .to(device),
                ).mean()
                + torch.mean(torch.abs(pred[:, :, 2]))
            )
            loss.backward()
            optim.step()
            logging.info(
                "train epoch %03i/%03i  batch %04i / %04i loss: %7.4f"
                % (epoch, args.max_epoch, id_b, len(tr_dl), loss.item())
            )
            epoch_loss += loss.item()
        # sched.step()
        log.add_scalar("Loss/train", epoch_loss / len(tr_dl), epoch)
        if epoch % args.val_interval == 0:
            with torch.no_grad():
                model.eval()

                val_loss = 0
                step = 0
                model.eval()
                gt = []
                pr = []
                inp_ = []
                dt = []

                for id_b, batch in enumerate(val_dl):
                    inp_.append(batch["src"])
                    gt.append(batch["trg"][:, :, 0:2])
                    dt.append(batch["dataset"])

                    seq_start_end = batch["seq_start_end"]
                    inp = (
                        batch["src"][:, 1:, 2:4].to(device) - mean.to(device)
                    ) / std.to(device)
                    start_of_seq = (
                        torch.Tensor([0, 0, 1])
                        .unsqueeze(0)
                        .unsqueeze(1)
                        .repeat(inp.shape[0], 1, 1)
                        .to(device)
                    )

                    dec_inp = start_of_seq
                    src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(device)
                    for i in range(args.preds):
                        trg_att = (
                            subsequent_mask(dec_inp.shape[1])
                            .repeat(dec_inp.shape[0], 1, 1)
                            .to(device)
                        )
                        out = model(inp, dec_inp, src_att, trg_att, seq_start_end)
                        dec_inp = torch.cat((dec_inp, out[:, -1:, :]), 1)

                    preds_tr_b = (
                        dec_inp[:, 1:, 0:2] * std.to(device) + mean.to(device)
                    ).cpu().numpy().cumsum(1) + batch["src"][:, -1:, 0:2].cpu().numpy()
                    pr.append(preds_tr_b)
                    logging.info(
                        "val epoch %03i/%03i  batch %04i / %04i"
                        % (epoch, args.max_epoch, id_b, len(val_dl))
                    )

                dt = np.concatenate(dt, 0)
                gt = np.concatenate(gt, 0)
                dt_names = test_dataset.data["dataset_name"]
                pr = np.concatenate(pr, 0)
                mad, fad, errs = baselineUtils.distance_metrics(gt, pr)
                log.add_scalar("validation/MAD", mad, epoch)
                log.add_scalar("validation/FAD", fad, epoch)

                if args.evaluate:

                    model.eval()
                    gt = []
                    pr = []
                    inp_ = []
                    dt = []

                    for id_b, batch in enumerate(test_dl):
                        inp_.append(batch["src"])
                        gt.append(batch["trg"][:, :, 0:2])
                        dt.append(batch["dataset"])

                        seq_start_end = batch["seq_start_end"]
                        inp = (
                            batch["src"][:, 1:, 2:4].to(device) - mean.to(device)
                        ) / std.to(device)

                        start_of_seq = (
                            torch.Tensor([0, 0, 1])
                            .unsqueeze(0)
                            .unsqueeze(1)
                            .repeat(inp.shape[0], 1, 1)
                            .to(device)
                        )

                        dec_inp = start_of_seq
                        src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(device)

                        for i in range(args.preds):
                            trg_att = (
                                subsequent_mask(dec_inp.shape[1])
                                .repeat(dec_inp.shape[0], 1, 1)
                                .to(device)
                            )
             
                            out = model(inp, dec_inp, src_att, trg_att, seq_start_end)
                            dec_inp = torch.cat((dec_inp, out[:, -1:, :]), 1)

                        preds_tr_b = (
                            dec_inp[:, 1:, 0:2] * std.to(device) + mean.to(device)
                        ).cpu().numpy().cumsum(1) + batch["src"][
                            :, -1:, 0:2
                        ].cpu().numpy()
                        pr.append(preds_tr_b)
                        logging.info(
                            "test epoch %03i/%03i  batch %04i / %04i"
                            % (epoch, args.max_epoch, id_b, len(test_dl))
                        )

                    dt = np.concatenate(dt, 0)
                    gt = np.concatenate(gt, 0)
                    dt_names = test_dataset.data["dataset_name"]
                    pr = np.concatenate(pr, 0)
                    mad, fad, errs = baselineUtils.distance_metrics(gt, pr)

                    log.add_scalar("eval/DET_mad", mad, epoch)
                    log.add_scalar("eval/DET_fad", fad, epoch)

                    # log.add_scalar('eval/DET_mad', mad, epoch)
                    # log.add_scalar('eval/DET_fad', fad, epoch)

                    scipy.io.savemat(
                        f"{model_output_dir}/det_{epoch}.mat",
                        {
                            "input": inp,
                            "gt": gt,
                            "pr": pr,
                            "dt": dt,
                            "dt_names": dt_names,
                        },
                    )

        if epoch % args.save_step == 0:

            torch.save(model.state_dict(), f"{checkpoint_dir}/{epoch:05d}.pth")

        epoch += 1
    ab = 1


if __name__ == "__main__":
    main()
