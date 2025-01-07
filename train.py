import argparse

import numpy as np
import pandas as pd
from accelerate import Accelerator
from Bio import SeqIO

from dnadiffusion.data.dataloader import load_data
from dnadiffusion.models.diffusion import Diffusion
from dnadiffusion.models.unet import UNet
from dnadiffusion.utils.train_util import TrainLoop


def train(data_path: str, data_type: str, batch_size: int, epochs: int, save_epoch: int, length: int, out_dir: str):
    accelerator = Accelerator(split_batches=True, log_with=["wandb"])
    accelerator.init_trackers(project_name="dnadiffusion", init_kwargs={"wandb": {"mode": "offline"}})

    # Ensure length is divisible by 8 for UNet
    length = length if length % 8 == 0 else (length // 8 + 1) * 8
    right_aligned = False

    if data_type == "mfe":
        right_aligned = True
        data = load_data_mfe(data_path=data_path, max_seq_len=length, right_aligned=right_aligned)
        num_classes = (10,)
    elif data_type == "mfe2":
        right_aligned = True
        data = load_data_mfe2(data_path=data_path, max_seq_len=length, right_aligned=right_aligned)
        num_classes = (10,)
    elif data_type == "rnaseq":
        right_aligned = True
        data = load_data_rnaseq(data_path=data_path, max_seq_len=length, right_aligned=right_aligned)
        num_classes = (10,)
    elif data_type == "rnaseq2":
        right_aligned = True
        data = load_data_rnaseq2(data_path=data_path, max_seq_len=length, right_aligned=right_aligned)
        num_classes = (10,)
    elif data_type == "te":
        right_aligned = True
        data = load_data_te(data_path=data_path, max_seq_len=length, right_aligned=right_aligned)
        num_classes = (10,)
    elif data_type == "rl":
        right_aligned = True
        data = load_data_rl(data_path=data_path, max_seq_len=length, right_aligned=right_aligned)
        num_classes = (10,)
    elif data_type == "deg":
        right_aligned = True
        data = load_data_deg(data_path=data_path, max_seq_len=length, right_aligned=right_aligned)
        num_classes = (10,)
    elif data_type == "deg2":
        right_aligned = True
        data = load_data_deg2(data_path=data_path, max_seq_len=length, right_aligned=right_aligned)
        num_classes = (10,)
    elif data_type == "rl_mfe":
        right_aligned = True
        data = load_data_rl_mfe(data_path=data_path, max_seq_len=length, right_aligned=right_aligned)
        num_classes = (10,10)
    else:
        msg = f"Invalid data type: {data_type}"
        raise ValueError(msg)

    unet = UNet(
        dim=length,
        channels=1,
        dim_mults=(1, 2, 4),
        resnet_block_groups=4,
        num_classes=num_classes
    )

    diffusion = Diffusion(
        unet,
        timesteps=50,
    )

    TrainLoop(
        data=data,
        model=diffusion,
        accelerator=accelerator,
        epochs=epochs,
        log_step_show=500,
        save_epoch=save_epoch,
        out_dir=out_dir,
        image_size=length,
        right_aligned=right_aligned,
        batch_size=batch_size,
    ).train_loop()


def load_data_mfe(
    data_path: str = "FiveSpecies_Cao_allutr_with_energy_structure.fasta",
    max_seq_len: int = 200,
    right_aligned: bool = False,
):
    # Preprocessing data
    mfes, seqs = [], []
    for record in SeqIO.parse(data_path, "fasta"):
        mfe = float(record.id.split("|")[0])
        mfes.append(mfe)
        seqs.append(str(record.seq))
    df = pd.DataFrame({"MFE": mfes, "sequence": seqs})
    df["len"] = df["sequence"].apply(len)
    df["AVG MFE"] = df["MFE"] / df["len"]
    df["TAG"] = "middle"
    am_mean = df["AVG MFE"].mean()
    am_std = df["AVG MFE"].std()
    df.loc[df["AVG MFE"]>am_mean+am_std, ["TAG"]] = "high"
    df.loc[df["AVG MFE"]<am_mean-am_std, ["TAG"]] = "low"

    return load_data(df, max_seq_len=max_seq_len, tag_name=["TAG"], right_aligned=right_aligned)


def load_data_mfe2(
    data_path: str,
    max_seq_len: int = 200,
    right_aligned: bool = False,
):
    # Preprocessing data
    df = pd.read_csv(data_path)
    df = pd.DataFrame({"mfe value": df["mfe"], "sequence": df["utr"]})
    df["len"] = df["sequence"].apply(len)
    df["AVG mfe"] = df["mfe value"] / df["len"]

    df["mfe"] = "middle"
    am_mean = df["AVG mfe"].mean()
    am_std = df["AVG mfe"].std()
    df.loc[df["AVG mfe"]>am_mean+am_std, ["mfe"]] = "high"
    df.loc[df["AVG mfe"]<am_mean-am_std, ["mfe"]] = "low"

    return load_data(df, max_seq_len=max_seq_len, tag_name=["mfe"], right_aligned=right_aligned)


def load_data_rnaseq(
    data_path: str = "./data/HEK_sequence.csv",
    max_seq_len: int = 200,
    right_aligned: bool = False,
):
    # Preprocessing data
    df = pd.read_csv(data_path)
    df = pd.DataFrame({"rnaseq_log": df["rnaseq_log"], "sequence": df["utr"]})
    df["sequence"] = df["sequence"].str.replace("<pad>", "")
    df["len"] = df["sequence"].apply(len)
    #df["AVG rl"] = df["rl"] / df["len"]
    df = df[df["len"]>=max_seq_len]
    df["TAG"] = "middle"
    df.loc[df["rnaseq_log"]<df["rnaseq_log"].quantile(0.33), ["TAG"]] = "low"
    df.loc[df["rnaseq_log"]>df["rnaseq_log"].quantile(0.66), ["TAG"]] = "high"

    return load_data(df, max_seq_len=max_seq_len, tag_name=["TAG"], right_aligned=right_aligned)

def load_data_rnaseq2(
    data_path: str,
    max_seq_len: int = 200,
    right_aligned: bool = False,
):
    # Preprocessing data
    df = pd.read_csv(data_path)
    df["rnaseq_log"] = df.loc[:, "rnaseq_HEK_fold0":"rnaseq_pc3_fold9"].mean(axis=1)
    df = pd.DataFrame({"rnaseq_log": df["rnaseq_log"], "sequence": df["utr"]})
    #df["sequence"] = df["sequence"].str.replace("<pad>", "")
    df["len"] = df["sequence"].apply(len)
    #df["AVG rl"] = df["rl"] / df["len"]
    #df = df[df["len"]>=max_seq_len]
    df["TAG"] = "middle"
    df.loc[df["rnaseq_log"]<df["rnaseq_log"].quantile(0.33), ["TAG"]] = "low"
    df.loc[df["rnaseq_log"]>df["rnaseq_log"].quantile(0.66), ["TAG"]] = "high"

    return load_data(df, max_seq_len=max_seq_len, tag_name=["TAG"], right_aligned=right_aligned)

def load_data_te(
    data_path: str = "./data/HEK_sequence.csv",
    max_seq_len: int = 200,
    right_aligned: bool = False,
):
    # Preprocessing data
    df = pd.read_csv(data_path)
    df = pd.DataFrame({"te_log": df["te_log"], "sequence": df["utr"]})
    df["sequence"] = df["sequence"].str.replace("<pad>", "")
    df["len"] = df["sequence"].apply(len)
    #df["AVG rl"] = df["rl"] / df["len"]
    df = df[df["len"]>=max_seq_len]
    df["TAG"] = "middle"
    te_mean = df["te_log"].mean()
    te_std = df["te_log"].std()
    df.loc[df["te_log"]<te_mean-te_std, ["TAG"]] = "low"
    df.loc[df["te_log"]>te_mean+te_std, ["TAG"]] = "high"

    return load_data(df, max_seq_len=max_seq_len, tag_name=["TAG"], right_aligned=right_aligned)


def load_data_rl(
    data_path: str = "./data/4.1_train_data_GSM3130435_egfp_unmod_1_BiologyFeatures.csv",
    max_seq_len: int = 200,
    right_aligned: bool = False,
):
    # Preprocessing data
    df = pd.read_csv(data_path)
    df = pd.DataFrame({"rl": df["rl"], "sequence": df["utr"]})
    df["len"] = df["sequence"].apply(len)
    df["AVG rl"] = df["rl"] / df["len"]
    df["TAG"] = "high"
    df.loc[df["rl"]<6.0, ["TAG"]] = "low"

    return load_data(df, max_seq_len=max_seq_len, tag_name=["TAG"], right_aligned=right_aligned)


def load_data_deg(
    data_path: str = "./data/train.json",
    max_seq_len: int = 200,
    right_aligned: bool = False,
    limit_total_sequences: int = 0,
    num_sampling_to_compare_cells: int = 1000,
):
    # Preprocessing data
    df = pd.read_json(data_path, lines=True)
    df["sum_deg"] = df["deg_pH10"].apply(sum)
    df["sequence"] = df["sequence"].str.slice(0, 68)
    df["sequence"] = df["sequence"].str.replace("U", "T")
    df = pd.DataFrame({"sum_deg": df["sum_deg"], "sequence": df["sequence"]})
    df["TAG"] = "middle"
    deg_mean = df["sum_deg"].mean()
    deg_std = df["sum_deg"].std()
    df.loc[df["sum_deg"]<deg_mean-deg_std, ["TAG"]] = "low"
    df.loc[df["sum_deg"]>deg_mean+deg_std, ["TAG"]] = "high"

    return load_data(df, max_seq_len=max_seq_len, tag_name=["TAG"], right_aligned=right_aligned)


def load_data_deg2(
    data_path: str = "./data/4.1_train_data_GSM3130435_egfp_unmod_1_BiologyFeatures.csv",
    max_seq_len: int = 200,
    right_aligned: bool = False,
):
    # Preprocessing data
    df = pd.read_csv(data_path)
    df = pd.DataFrame({"deg": df["deg"], "sequence": df["utr"]})
    df["TAG"] = "middle"
    deg_mean = df["deg"].mean()
    deg_std = df["deg"].std()
    df.loc[df["deg"]<deg_mean-deg_std, ["TAG"]] = "low"
    df.loc[df["deg"]>deg_mean+deg_std, ["TAG"]] = "high"

    return load_data(df, max_seq_len=max_seq_len, tag_name=["TAG"], right_aligned=right_aligned)


def load_data_rl_mfe(
    data_path: str,
    max_seq_len: int = 200,
    right_aligned: bool = False,
):
    # Preprocessing data
    df = pd.read_csv(data_path)
    df = pd.DataFrame({"rl value": df["rl"], "mfe value": df["mfe"], "sequence": df["utr"]})
    df["len"] = df["sequence"].apply(len)
    df["AVG rl"] = df["rl value"] / df["len"]
    df["AVG mfe"] = df["mfe value"] / df["len"]

    df["rl"] = "high"
    df.loc[df["rl value"]<6.0, ["rl"]] = "low"

    df["mfe"] = "middle"
    am_mean = df["AVG mfe"].mean()
    am_std = df["AVG mfe"].std()
    df.loc[df["AVG mfe"]>am_mean+am_std, ["mfe"]] = "high"
    df.loc[df["AVG mfe"]<am_mean-am_std, ["mfe"]] = "low"

    return load_data(df, max_seq_len=max_seq_len, tag_name=["rl", "mfe"], right_aligned=right_aligned)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dnadiffusion")
    parser.add_argument("--input", type=str, help="input file")
    parser.add_argument("--output", type=str, default="checkpoints", help="output dir")
    parser.add_argument("--type", type=str, choices=["mfe", "mfe2", "rnaseq", "rnaseq2", "te", "rl", "deg", "deg2", "rl_mfe"], help="data type")
    parser.add_argument("--batch-size", type=int, default=480, help="batch size")
    parser.add_argument("--epochs", type=int, default=10000, help="the number of epochs")
    parser.add_argument("--save-epoch", type=int, default=500, help="the interval of saving the model")
    parser.add_argument("--length", type=int, default=200, help="sequence length")
    args = parser.parse_args()

    train(
        data_path=args.input,
        out_dir=args.output,
        data_type=args.type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        save_epoch=args.save_epoch,
        length=args.length)
