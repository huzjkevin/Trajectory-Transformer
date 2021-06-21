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
import random

from torch.utils.tensorboard import SummaryWriter
import quantized_TF


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
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--delim", type=str, default="\t")
    parser.add_argument("--name", type=str, default="trajectory_combined")
    parser.add_argument("--epoch", type=str, default="00000")
    parser.add_argument("--num_samples", type=int, default="20")

    args = parser.parse_args()
    model_name = args.name

    # try:
    #     os.mkdir("models")
    # except:
    #     pass
    # try:
    #     os.mkdir("output")
    # except:
    #     pass
    # try:
    #     os.mkdir("output/IndividualTF")
    # except:
    #     pass
    # try:
    #     os.mkdir(f"models/IndividualTF")
    # except:
    #     pass

    # try:
    #     os.mkdir(f"output/IndividualTF/{args.name}")
    # except:
    #     pass

    # try:
    #     os.mkdir(f"models/IndividualTF/{args.name}")
    # except:
    #     pass
    seed = 72
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda")

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")

    args.verbose = True

    ## creation of the dataloaders for train and validation

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

    mat = scipy.io.loadmat(
        os.path.join("models/IndividualTF", args.dataset_name, "norm.mat")
    )

    mean = torch.from_numpy(mat["mean"])
    std = torch.from_numpy(mat["std"])

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

    checkpoint_dir = os.path.join(
        "exp_trajectory_combined_20210621113252", f"IndividualTF_ckpts"
    )
    model.load_state_dict(
        torch.load(f"{checkpoint_dir}/{args.epoch}.pth")
    )
    model.to(device)

    test_dl = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
        collate_fn=baselineUtils.collate_fn,
    )

    # DETERMINISTIC MODE
    with torch.no_grad():
        model.eval()
        gt = []
        pr = []
        inp_ = []
        dt = []

        ade, fde = [], []
        for id_b, batch in enumerate(test_dl):
            print(f"batch {id_b:03d}/{len(test_dl)}")
            
            dt.append(batch["dataset"])
            seq_start_end = batch["seq_start_end"]
            _gt = batch["trg"][:, :, 0:2]
            gt.append(_gt)
            inp = (batch["src"][:, 1:, 2:4].to(device) - mean.to(device)) / std.to(
                device
            )

            src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(device)
            start_of_seq = (
                torch.Tensor([0, 0, 1])
                .unsqueeze(0)
                .unsqueeze(1)
                .repeat(inp.shape[0], 1, 1)
                .to(device)
            )
            dec_inp = start_of_seq

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

            _mad, _fad, _ = baselineUtils.distance_metrics(_gt, preds_tr_b)
            ade.append(_mad)
            fde.append(_fad)

            pr.append(preds_tr_b)

        dt = np.concatenate(dt, 0)
        gt = np.concatenate(gt, 0)
        dt_names = test_dataset.data["dataset_name"]
        pr = np.concatenate(pr, 0)
        # mad, fad, errs = baselineUtils.distance_metrics(gt, pr)

        ade = np.array(ade)
        fde = np.array(fde)
        mad = ade.mean()
        fad = fde.mean()
        scipy.io.savemat(
            f"output/IndividualTF/{args.name}/MM_deterministic.mat",
            {
                "input": inp,
                "gt": gt,
                "pr": pr,
                "dt": dt,
                "dt_names": dt_names,
            },
        )

        print("Determinitic:")
        print("mad: %6.3f" % mad)
        print("fad: %6.3f" % fad)

        # MULTI MODALITY
        num_samples = args.num_samples

        model.eval()
        gt = []
        pr_all = {}
        inp_ = []
        dt = []
        ade, fde = [], []
        for sam in range(num_samples):
            pr_all[sam] = []
        for id_b, batch in enumerate(test_dl):
            print(f"batch {id_b:03d}/{len(test_dl)}")

            seq_start_end = batch["seq_start_end"]
            dt.append(batch["dataset"])
            _gt = batch["trg"][:, :, 0:2]
            gt.append(_gt)
            inp = (batch["src"][:, 1:, 2:4].to(device) - mean.to(device)) / std.to(
                device
            )

            src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(device)
            start_of_seq = (
                torch.Tensor([0, 0, 1])
                .unsqueeze(0)
                .unsqueeze(1)
                .repeat(inp.shape[0], 1, 1)
                .to(device)
            )

            ade_sample, fde_sample = [], []
            for sam in range(num_samples):
                dec_inp = start_of_seq
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

                _mad, _fad, _ = baselineUtils.distance_metrics(_gt, preds_tr_b)
                ade_sample.append(_mad)
                fde_sample.append(_fad)

                # pr_all[sam].append(preds_tr_b)
            
            ade.append(min(ade_sample))
            fde.append(min(fde_sample))

        dt = np.concatenate(dt, 0)
        gt = np.concatenate(gt, 0)
        dt_names = test_dataset.data["dataset_name"]

        # samp = {}
        # for k in pr_all.keys():
        #     samp[k] = {}
        #     samp[k]["pr"] = np.concatenate(pr_all[k], 0)
        #     (
        #         samp[k]["mad"],
        #         samp[k]["fad"],
        #         samp[k]["err"],
        #     ) = baselineUtils.distance_metrics(gt, samp[k]["pr"])

        # ev = [samp[i]["err"] for i in range(num_samples)]
        # e20 = np.stack(ev, -1)
        # mad_samp = e20.mean(1).min(-1).mean()
        # fad_samp = e20[:, -1].min(-1).mean()
        # preds_all_fin = np.stack(list([samp[i]["pr"] for i in range(num_samples)]), -1)

        ade = np.array(ade)
        fde = np.array(fde)
        mad_samp = ade.mean()
        fad_samp = fde.mean()

        scipy.io.savemat(
            f"output/IndividualTF/{args.name}/MM_{num_samples}.mat",
            {
                "input": inp,
                "gt": gt,
                # "pr": preds_all_fin,
                "dt": dt,
                "dt_names": dt_names,
            },
        )

        print("Determinitic:")
        print("mad: %6.3f" % mad)
        print("fad: %6.3f" % fad)

        print("Multimodality:")
        print("mad: %6.3f" % mad_samp)
        print("fad: %6.3f" % fad_samp)


if __name__ == "__main__":
    main()
