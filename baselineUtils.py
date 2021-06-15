from numpy.lib.utils import info
from torch._C import dtype
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import torch
import random
import scipy.spatial
import scipy.io
import math


def create_dataset(
    dataset_folder,
    dataset_name,
    val_size,
    gt,
    horizon,
    delim="\t",
    train=True,
    eval=False,
    verbose=False,
):

    if train == True:
        datasets_list = os.listdir(os.path.join(dataset_folder, dataset_name, "train"))
        full_dt_folder = os.path.join(dataset_folder, dataset_name, "train")
    if train == False and eval == False:
        datasets_list = os.listdir(os.path.join(dataset_folder, dataset_name, "val"))
        full_dt_folder = os.path.join(dataset_folder, dataset_name, "val")
    if train == False and eval == True:
        datasets_list = os.listdir(os.path.join(dataset_folder, dataset_name, "test"))
        full_dt_folder = os.path.join(dataset_folder, dataset_name, "test")

    datasets_list = datasets_list
    data = {}
    data_src = []
    data_trg = []
    data_seq_start = []
    data_frames = []
    data_dt = []
    data_peds = []
    seq_start_end = []
    num_peds_in_seq = []
    num_seq = 0

    val_src = []
    val_trg = []
    val_seq_start = []
    val_frames = []
    val_dt = []
    val_peds = []

    if verbose:
        print("start loading dataset")
        print("validation set size -> %i" % (val_size))

    for i_dt, dt in enumerate(datasets_list):
        if verbose:
            print("%03i / %03i - loading %s" % (i_dt + 1, len(datasets_list), dt))
        raw_data = pd.read_csv(
            os.path.join(full_dt_folder, dt),
            delimiter=delim,
            names=["frame", "ped", "x", "y"],
            usecols=[0, 1, 2, 3],
            na_values="?",
        )

        raw_data.sort_values(by=["frame", "ped"], inplace=True)
        raw_data = raw_data.to_numpy()

        info = get_scene_sequence(raw_data, gt + horizon, 1)
        inp = info["inp_norm"][:, :gt]
        out = info["inp_norm"][:, gt:]

        dt_seq_start = info["seq_start"]
        dt_dataset = np.array([i_dt]).repeat(inp.shape[0])

        data_src.append(inp)
        data_trg.append(out)
        data_seq_start.append(dt_seq_start)
        data_dt.append(dt_dataset)
        num_peds_in_seq += info["num_peds_in_seq"]
        num_seq += info["num_seq"]
    
    cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
    seq_start_end = [
        (start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])
    ]
    data["src"] = np.concatenate(data_src, 0)
    data["trg"] = np.concatenate(data_trg, 0)
    data["seq_start"] = np.concatenate(data_seq_start, 0)
    data["dataset"] = np.concatenate(data_dt, 0)
    data["dataset_name"] = datasets_list
    data["seq_start_end"] = seq_start_end
    data["num_seq"] = num_seq

    mean = data["src"].mean((0, 1))
    std = data["src"].std((0, 1))

    return IndividualTfDataset(data, "train", mean, std), None

    return IndividualTfDataset(data, "train", mean, std), IndividualTfDataset(
        data_val, "validation", mean, std
    )


def collate_fn(data):
    # {
    #     "src": torch.Tensor(self.data["src"][start: end]),
    #     "trg": torch.Tensor(self.data["trg"][start: end]),
    #     # "frames": self.data["frames"][index],
    #     "seq_start": self.data["seq_start"][start: end],
    #     "dataset": self.data["dataset"][start: end],
    #     # "peds": self.data["peds"][index],
    # }
    num_peds = []
    src = []
    trg = []
    seq_start = []
    dataset = []
    for sample in data:
        src.append(sample["src"])
        trg.append(sample["trg"])
        seq_start.append(sample["seq_start"])
        dataset.append(sample["dataset"])
        num_peds.append(len(sample["src"]))
    cum_start_idx = [0] + np.cumsum(num_peds).tolist()
    seq_start_end = [
        [start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])
    ]
    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    src = torch.cat(src, dim=0)
    trg = torch.cat(trg, dim=0)
    # seq_start = torch.cat(seq_start, dim=0)
    # dataset = torch.cat(dataset, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    batch = {
        "src": src,
        "trg": trg,
        "seq_start": seq_start,
        "dataset": dataset,
        "seq_start_end": seq_start_end,
    }

    return batch

class IndividualTfDataset(Dataset):
    def __init__(self, data, name, mean, std):
        super(IndividualTfDataset, self).__init__()

        self.data = data
        self.name = name

        self.mean = mean
        self.std = std

    def __len__(self):
        return self.data["num_seq"]

    def __getitem__(self, index):
        start, end = self.data["seq_start_end"][index]
        return {
            "src": torch.from_numpy(self.data["src"][start: end]),
            "trg": torch.from_numpy(self.data["trg"][start: end]),
            "seq_start": self.data["seq_start"][start: end],
            "dataset": self.data["dataset"][start: end],
        }


def create_folders(baseFolder, datasetName):
    try:
        os.mkdir(baseFolder)
    except:
        pass

    try:
        os.mkdir(os.path.join(baseFolder, datasetName))
    except:
        pass


def get_strided_data(dt, gt_size, horizon, step):
    inp_te = []
    dtt = dt.astype(np.float32)
    raw_data = dtt

    ped = raw_data.ped.unique()
    frame = []
    ped_ids = []
    for p in ped:
        for i in range(
            1 + (raw_data[raw_data.ped == p].shape[0] - gt_size - horizon) // step
        ):
            frame.append(
                dt[dt.ped == p]
                .iloc[i * step : i * step + gt_size + horizon, [0]]
                .values.squeeze()
            )
            # print("%i,%i,%i" % (i * 4, i * 4 + gt_size, i * 4 + gt_size + horizon))
            inp_te.append(
                raw_data[raw_data.ped == p]
                .iloc[i * step : i * step + gt_size + horizon, 2:4]
                .values
            )
            ped_ids.append(p)

    frames = np.stack(frame)
    inp_te_np = np.stack(inp_te)
    ped_ids = np.stack(ped_ids)

    inp_no_start = inp_te_np[:, 1:, 0:2] - inp_te_np[:, :-1, 0:2]
    inp_std = inp_no_start.std(axis=(0, 1))
    inp_mean = inp_no_start.mean(axis=(0, 1))
    inp_norm = inp_no_start
    # inp_norm = (inp_no_start - inp_mean) / inp_std

    # vis=inp_te_np[:,1:,2:4]/np.linalg.norm(inp_te_np[:,1:,2:4],2,axis=2)[:,:,np.newaxis]
    # inp_norm=np.concatenate((inp_norm,vis),2)

    return (
        inp_norm[:, : gt_size - 1],
        inp_norm[:, gt_size - 1 :],
        {
            "mean": inp_mean,
            "std": inp_std,
            "seq_start": inp_te_np[:, 0:1, :].copy(),
            "frames": frames,
            "peds": ped_ids,
        },
    )


def get_strided_data_2(dt, gt_size, horizon, step):
    inp_te = []
    dtt = dt.astype(np.float32)
    raw_data = dtt

    ped = raw_data.ped.unique()
    frame = []
    ped_ids = []
    for p in ped:
        for i in range(
            1 + (raw_data[raw_data.ped == p].shape[0] - gt_size - horizon) // step
        ):
            frame.append(
                dt[dt.ped == p]
                .iloc[i * step : i * step + gt_size + horizon, [0]]
                .values.squeeze()
            )
            # print("%i,%i,%i" % (i * 4, i * 4 + gt_size, i * 4 + gt_size + horizon))
            inp_te.append(
                raw_data[raw_data.ped == p]
                .iloc[i * step : i * step + gt_size + horizon, 2:4]
                .values
            )
            ped_ids.append(p)

    frames = np.stack(frame)
    inp_te_np = np.stack(inp_te)
    ped_ids = np.stack(ped_ids)

    inp_relative_pos = inp_te_np - inp_te_np[:, :1, :]
    inp_speed = np.concatenate(
        (
            np.zeros((inp_te_np.shape[0], 1, 2)),
            inp_te_np[:, 1:, 0:2] - inp_te_np[:, :-1, 0:2],
        ),
        1,
    )
    inp_accel = np.concatenate(
        (
            np.zeros((inp_te_np.shape[0], 1, 2)),
            inp_speed[:, 1:, 0:2] - inp_speed[:, :-1, 0:2],
        ),
        1,
    )
    # inp_std = inp_no_start.std(axis=(0, 1))
    # inp_mean = inp_no_start.mean(axis=(0, 1))
    # inp_norm= inp_no_start
    # inp_norm = (inp_no_start - inp_mean) / inp_std

    # vis=inp_te_np[:,1:,2:4]/np.linalg.norm(inp_te_np[:,1:,2:4],2,axis=2)[:,:,np.newaxis]
    # inp_norm=np.concatenate((inp_norm,vis),2)
    inp_norm = np.concatenate((inp_te_np, inp_relative_pos, inp_speed, inp_accel), 2)
    inp_mean = np.zeros(8)
    inp_std = np.ones(8)

    return (
        inp_norm[:, :gt_size],
        inp_norm[:, gt_size:],
        {
            "mean": inp_mean,
            "std": inp_std,
            "seq_start": inp_te_np[:, 0:1, :].copy(),
            "frames": frames,
            "peds": ped_ids,
        },
    )


def get_strided_data_clust(dt, gt_size, horizon, step):
    inp_te = []
    dtt = dt.astype(np.float32)
    raw_data = dtt

    ped = raw_data.ped.unique()
    frame = []
    ped_ids = []
    for p in ped:
        for i in range(
            1 + (raw_data[raw_data.ped == p].shape[0] - gt_size - horizon) // step
        ):
            frame.append(
                dt[dt.ped == p]
                .iloc[i * step : i * step + gt_size + horizon, [0]]
                .values.squeeze()
            )
            # print("%i,%i,%i" % (i * 4, i * 4 + gt_size, i * 4 + gt_size + horizon))
            inp_te.append(
                raw_data[raw_data.ped == p]
                .iloc[i * step : i * step + gt_size + horizon, 2:4]
                .values
            )
            ped_ids.append(p)

    frames = np.stack(frame)
    inp_te_np = np.stack(inp_te)
    ped_ids = np.stack(ped_ids)

    # inp_relative_pos= inp_te_np-inp_te_np[:,:1,:]
    inp_speed = np.concatenate(
        (
            np.zeros((inp_te_np.shape[0], 1, 2)),
            inp_te_np[:, 1:, 0:2] - inp_te_np[:, :-1, 0:2],
        ),
        1,
    )
    # inp_accel = np.concatenate((np.zeros((inp_te_np.shape[0],1,2)),inp_speed[:,1:,0:2] - inp_speed[:, :-1, 0:2]),1)
    # inp_std = inp_no_start.std(axis=(0, 1))
    # inp_mean = inp_no_start.mean(axis=(0, 1))
    # inp_norm= inp_no_start
    # inp_norm = (inp_no_start - inp_mean) / inp_std

    # vis=inp_te_np[:,1:,2:4]/np.linalg.norm(inp_te_np[:,1:,2:4],2,axis=2)[:,:,np.newaxis]
    # inp_norm=np.concatenate((inp_norm,vis),2)
    inp_norm = np.concatenate((inp_te_np, inp_speed), 2)
    inp_mean = np.zeros(4)
    inp_std = np.ones(4)

    return (
        inp_norm[:, :gt_size],
        inp_norm[:, gt_size:],
        {
            "mean": inp_mean,
            "std": inp_std,
            "seq_start": inp_te_np[:, 0:1, :].copy(),
            "frames": frames,
            "peds": ped_ids,
        },
    )


def get_scene_sequence(data, seq_len, skip, min_ped=1):
    data = data.astype(np.float32)
    frames = np.unique(data[:, 0]).tolist()
    frame_data = []
    for frame in frames:
        frame_data.append(data[frame == data[:, 0], :])
    num_sequences = int(math.ceil((len(frames) - seq_len + 1) / skip))

    num_peds_in_seq = []
    seq_list = []
    seq_list_rel = []
    info = {}

    for idx in range(0, num_sequences * skip + 1, skip):
        # curr_seq_data is a 20 length sequence
        curr_seq_data = np.concatenate(frame_data[idx : idx + seq_len], axis=0)
        peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
        curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, seq_len), dtype=np.float32)
        curr_seq = np.zeros((len(peds_in_curr_seq), 2, seq_len), dtype=np.float32)
        num_peds_considered = 0
        _non_linear_ped = []

        for _, ped_id in enumerate(peds_in_curr_seq):
            # a = curr_seq_data[:, 1] == ped_id
            curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
            curr_ped_seq = np.around(curr_ped_seq, decimals=4)
            pad_front = frames.index(curr_ped_seq[0, 0]) - idx
            pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1

            if pad_end - pad_front != seq_len:
                continue
            curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
            curr_ped_seq = curr_ped_seq
            # Make coordinates relative
            rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
            rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
            _idx = num_peds_considered
            curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
            curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
            # Linear vs Non-Linear Trajectory
            num_peds_considered += 1

        if num_peds_considered > min_ped:
            num_peds_in_seq.append(num_peds_considered)
            seq_list.append(curr_seq[:num_peds_considered])
            seq_list_rel.append(curr_seq_rel[:num_peds_considered])

    # num_peds_in_seq = np.concatenate(num_peds_in_seq, 0)
    num_seq = len(seq_list)
    seq_list = np.concatenate(seq_list, 0).transpose(0, 2, 1)
    seq_list_rel = np.concatenate(seq_list_rel, 0).transpose(0, 2, 1)
    inp_norm = np.concatenate((seq_list, seq_list_rel), 2)
    inp_mean = np.zeros(4)
    inp_std = np.ones(4)

    info["num_peds_in_seq"] = num_peds_in_seq
    info["inp_norm"] = inp_norm
    info["inp_mean"] = inp_mean
    info["inp_std"] = inp_std
    info["seq_start"] = seq_list[:, 0:1, :].copy()
    info["num_seq"] = num_seq

    return info


def distance_metrics(gt, preds):
    errors = np.zeros(preds.shape[:-1])
    for i in range(errors.shape[0]):
        for j in range(errors.shape[1]):
            errors[i, j] = scipy.spatial.distance.euclidean(gt[i, j], preds[i, j])
    return errors.mean(), errors[:, -1].mean(), errors
