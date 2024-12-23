import os
import pickle
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from Bio import SeqIO
from torch.utils.data import DataLoader, Dataset

from dnadiffusion.utils.utils import one_hot_encode_r


def load_data_mfe(
    data_path: str = "FiveSpecies_Cao_allutr_with_energy_structure.fasta",
    max_seq_len: int = 200,
    limit_total_sequences: int = 0,
    num_sampling_to_compare_cells: int = 1000,
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

    # Creating sequence dataset
    nucleotides = ["A", "C", "G", "T"]
    x_train_seq = np.array([one_hot_encode_r(x, nucleotides, max_seq_len=max_seq_len) for x in df["sequence"] if "N" not in x])
    X_train = np.array([x.T.tolist() for x in x_train_seq])
    X_train[X_train == 0] = -1

    # Creating labels
    tag_to_numeric = {x: n for n, x in enumerate(df["TAG"].unique(), 1)}
    numeric_to_tag = dict(enumerate(df["TAG"].unique(), 1))
    cell_types = list(numeric_to_tag.keys())
    x_train_cell_type = torch.tensor([tag_to_numeric[x] for x in df["TAG"]])

    # Collecting variables into a dict
    encode_data_dict = {
        # "train_motifs": train_motifs,
        # "train_motifs_cell_specific": train_motifs_cell_specific,
        # "test_motifs": test_motifs,
        # "test_motifs_cell_specific": test_motifs_cell_specific,
        # "shuffle_motifs": shuffle_motifs,
        # "shuffle_motifs_cell_specific": shuffle_motifs_cell_specific,
        "tag_to_numeric": tag_to_numeric,
        "numeric_to_tag": numeric_to_tag,
        "cell_types": cell_types,
        "X_train": X_train,
        "x_train_cell_type": x_train_cell_type,
    }

    return encode_data_dict


def load_data_rnaseq(
    data_path: str = "./data/HEK_sequence.csv",
    max_seq_len: int = 200,
    limit_total_sequences: int = 0,
    num_sampling_to_compare_cells: int = 1000,
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

    # Creating sequence dataset
    nucleotides = ["A", "C", "G", "T"]
    x_train_seq = np.array([one_hot_encode_r(x, nucleotides, max_seq_len=max_seq_len) for x in df["sequence"] if "N" not in x])
    X_train = np.array([x.T.tolist() for x in x_train_seq])
    X_train[X_train == 0] = -1

    # Creating labels
    tag_to_numeric = {x: n for n, x in enumerate(df["TAG"].unique(), 1)}
    numeric_to_tag = dict(enumerate(df["TAG"].unique(), 1))
    cell_types = list(numeric_to_tag.keys())
    x_train_cell_type = torch.tensor([tag_to_numeric[x] for x in df["TAG"]])

    # Collecting variables into a dict
    encode_data_dict = {
        # "train_motifs": train_motifs,
        # "train_motifs_cell_specific": train_motifs_cell_specific,
        # "test_motifs": test_motifs,
        # "test_motifs_cell_specific": test_motifs_cell_specific,
        # "shuffle_motifs": shuffle_motifs,
        # "shuffle_motifs_cell_specific": shuffle_motifs_cell_specific,
        "tag_to_numeric": tag_to_numeric,
        "numeric_to_tag": numeric_to_tag,
        "cell_types": cell_types,
        "X_train": X_train,
        "x_train_cell_type": x_train_cell_type,
    }

    return encode_data_dict


def load_data_te(
    data_path: str = "./data/HEK_sequence.csv",
    max_seq_len: int = 200,
    limit_total_sequences: int = 0,
    num_sampling_to_compare_cells: int = 1000,
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

    # Creating sequence dataset
    nucleotides = ["A", "C", "G", "T"]
    x_train_seq = np.array([one_hot_encode_r(x, nucleotides, max_seq_len=max_seq_len) for x in df["sequence"] if "N" not in x])
    X_train = np.array([x.T.tolist() for x in x_train_seq])
    X_train[X_train == 0] = -1

    # Creating labels
    tag_to_numeric = {x: n for n, x in enumerate(df["TAG"].unique(), 1)}
    numeric_to_tag = dict(enumerate(df["TAG"].unique(), 1))
    cell_types = list(numeric_to_tag.keys())
    x_train_cell_type = torch.tensor([tag_to_numeric[x] for x in df["TAG"]])

    # Collecting variables into a dict
    encode_data_dict = {
        # "train_motifs": train_motifs,
        # "train_motifs_cell_specific": train_motifs_cell_specific,
        # "test_motifs": test_motifs,
        # "test_motifs_cell_specific": test_motifs_cell_specific,
        # "shuffle_motifs": shuffle_motifs,
        # "shuffle_motifs_cell_specific": shuffle_motifs_cell_specific,
        "tag_to_numeric": tag_to_numeric,
        "numeric_to_tag": numeric_to_tag,
        "cell_types": cell_types,
        "X_train": X_train,
        "x_train_cell_type": x_train_cell_type,
    }

    return encode_data_dict


def load_data_rl(
    data_path: str = "./data/4.1_train_data_GSM3130435_egfp_unmod_1_BiologyFeatures.csv",
    max_seq_len: int = 200,
    limit_total_sequences: int = 0,
    num_sampling_to_compare_cells: int = 1000,
):
    # Preprocessing data
    df = pd.read_csv(data_path)
    df = pd.DataFrame({"rl": df["rl"], "sequence": df["utr"]})
    df["len"] = df["sequence"].apply(len)
    df["AVG rl"] = df["rl"] / df["len"]
    df["TAG"] = "high"
    df.loc[df["rl"]<6.0, ["TAG"]] = "low"

    # Creating sequence dataset
    nucleotides = ["A", "C", "G", "T"]
    x_train_seq = np.array([one_hot_encode_r(x, nucleotides, max_seq_len=max_seq_len) for x in df["sequence"] if "N" not in x])
    X_train = np.array([x.T.tolist() for x in x_train_seq])
    X_train[X_train == 0] = -1

    # Creating labels
    tag_to_numeric = {x: n for n, x in enumerate(df["TAG"].unique(), 1)}
    numeric_to_tag = dict(enumerate(df["TAG"].unique(), 1))
    cell_types = list(numeric_to_tag.keys())
    x_train_cell_type = torch.tensor([tag_to_numeric[x] for x in df["TAG"]])

    # Collecting variables into a dict
    encode_data_dict = {
        # "train_motifs": train_motifs,
        # "train_motifs_cell_specific": train_motifs_cell_specific,
        # "test_motifs": test_motifs,
        # "test_motifs_cell_specific": test_motifs_cell_specific,
        # "shuffle_motifs": shuffle_motifs,
        # "shuffle_motifs_cell_specific": shuffle_motifs_cell_specific,
        "tag_to_numeric": tag_to_numeric,
        "numeric_to_tag": numeric_to_tag,
        "cell_types": cell_types,
        "X_train": X_train,
        "x_train_cell_type": x_train_cell_type,
    }

    return encode_data_dict


def load_data_deg(
    data_path: str = "./data/train.json",
    max_seq_len: int = 200,
    limit_total_sequences: int = 0,
    num_sampling_to_compare_cells: int = 1000,
):
    # Preprocessing data
    df = pd.read_json("./data/train.json", lines=True)
    df["sum_deg"] = df["deg_pH10"].apply(sum)
    df["sequence"] = df["sequence"].str.slice(0, 68)
    df["sequence"] = df["sequence"].str.replace("U", "T")
    df = pd.DataFrame({"sum_deg": df["sum_deg"], "sequence": df["sequence"]})
    df["TAG"] = "middle"
    deg_mean = df["sum_deg"].mean()
    deg_std = df["sum_deg"].std()
    df.loc[df["sum_deg"]<deg_mean-deg_std, ["TAG"]] = "low"
    df.loc[df["sum_deg"]>deg_mean+deg_std, ["TAG"]] = "high"

    # Creating sequence dataset
    nucleotides = ["A", "C", "G", "T"]
    x_train_seq = np.array([one_hot_encode_r(x, nucleotides, max_seq_len=max_seq_len) for x in df["sequence"] if "N" not in x])
    X_train = np.array([x.T.tolist() for x in x_train_seq])
    X_train[X_train == 0] = -1

    # Creating labels
    tag_to_numeric = {x: n for n, x in enumerate(df["TAG"].unique(), 1)}
    numeric_to_tag = dict(enumerate(df["TAG"].unique(), 1))
    cell_types = list(numeric_to_tag.keys())
    x_train_cell_type = torch.tensor([tag_to_numeric[x] for x in df["TAG"]])

    # Collecting variables into a dict
    encode_data_dict = {
        # "train_motifs": train_motifs,
        # "train_motifs_cell_specific": train_motifs_cell_specific,
        # "test_motifs": test_motifs,
        # "test_motifs_cell_specific": test_motifs_cell_specific,
        # "shuffle_motifs": shuffle_motifs,
        # "shuffle_motifs_cell_specific": shuffle_motifs_cell_specific,
        "tag_to_numeric": tag_to_numeric,
        "numeric_to_tag": numeric_to_tag,
        "cell_types": cell_types,
        "X_train": X_train,
        "x_train_cell_type": x_train_cell_type,
    }

    return encode_data_dict


def motifs_from_fasta(fasta: str):
    print("Computing Motifs....")
    os.system(f"gimme scan {fasta} -p  JASPAR2020_vertebrates -g hg38 -n 20> train_results_motifs.bed")
    df_results_seq_guime = pd.read_csv("train_results_motifs.bed", sep="\t", skiprows=5, header=None)
    df_results_seq_guime["motifs"] = df_results_seq_guime[8].apply(lambda x: x.split('motif_name "')[1].split('"')[0])

    df_results_seq_guime[0] = df_results_seq_guime[0].apply(lambda x: "_".join(x.split("_")[:-1]))
    df_results_seq_guime_count_out = df_results_seq_guime[[0, "motifs"]].groupby("motifs").count()
    return df_results_seq_guime_count_out


def save_fasta(df: pd.DataFrame, name: str, num_sequences: int, seq_to_subset_comp: bool = False) -> str:
    fasta_path = f"{name}.fasta"
    save_fasta_file = open(fasta_path, "w")
    num_to_sample = df.shape[0]

    # Subsetting sequences
    if num_sequences and seq_to_subset_comp:
        num_to_sample = num_sequences

    # Sampling sequences
    print(f"Sampling {num_to_sample} sequences")
    write_fasta_component = "\n".join(
        df[["dhs_id", "sequence", "TAG"]]
        .head(num_to_sample)
        .apply(lambda x: f">{x[0]}_TAG_{x[2]}\n{x[1]}", axis=1)
        .values.tolist()
    )
    save_fasta_file.write(write_fasta_component)
    save_fasta_file.close()

    return fasta_path


def generate_motifs_and_fastas(
    df: pd.DataFrame, name: str, num_sequences: int, subset_list: list | None = None
) -> dict[str, Any]:
    print("Generating Motifs and Fastas...", name)
    print("---" * 10)

    # Saving fasta
    if subset_list:
        fasta_path = save_fasta(df, f"{name}_{'_'.join([str(c) for c in subset_list])}", num_sequences)
    else:
        fasta_path = save_fasta(df, name, num_sequences)

    # Computing motifs
    motifs = motifs_from_fasta(fasta_path)

    # Generating subset specific motifs
    final_subset_motifs = {}
    for comp, v_comp in df.groupby("TAG"):
        print(comp)
        c_fasta = save_fasta(v_comp, f"{name}_{comp}", num_sequences, seq_to_subset_comp=True)
        final_subset_motifs[comp] = motifs_from_fasta(c_fasta)

    return {
        "fasta_path": fasta_path,
        "motifs": motifs,
        "final_subset_motifs": final_subset_motifs,
        "df": df,
    }


def preprocess_data(
    input_csv: str,
    subset_list: list | None = None,
    limit_total_sequences: int | None = None,
    number_of_sequences_to_motif_creation: int = 1000,
    save_output: bool = True,
):
    # Reading the csv file
    df = pd.read_csv(input_csv, sep="\t")

    # Subsetting the dataframe
    if subset_list:
        print(" or ".join([f"TAG == {c}" for c in subset_list]))
        df = df.query(" or ".join([f'TAG == "{c}" ' for c in subset_list]))
        print("Subsetting...")

    # Limiting the total number of sequences
    if limit_total_sequences > 0:
        print(f"Limiting total sequences to {limit_total_sequences}")
        df = df.sample(limit_total_sequences)

    # Creating train/test/shuffle groups
    df_test = df[df["chr"] == "chr1"].reset_index(drop=True)
    df_train_shuffled = df[df["chr"] == "chr2"].reset_index(drop=True)
    df_train = df_train = df[(df["chr"] != "chr1") & (df["chr"] != "chr2")].reset_index(drop=True)

    df_train_shuffled["sequence"] = df_train_shuffled["sequence"].apply(
        lambda x: "".join(random.sample(list(x), len(x)))
    )

    # Getting motif information from the sequences
    train = generate_motifs_and_fastas(df_train, "train", number_of_sequences_to_motif_creation, subset_list)
    test = generate_motifs_and_fastas(df_test, "test", number_of_sequences_to_motif_creation, subset_list)
    train_shuffled = generate_motifs_and_fastas(
        df_train_shuffled,
        "train_shuffled",
        number_of_sequences_to_motif_creation,
        subset_list,
    )

    combined_dict = {"train": train, "test": test, "train_shuffled": train_shuffled}

    # Writing to pickle
    if save_output:
        # Saving all train, test, train_shuffled dictionaries to pickle
        with open("src/dnadiffusion/data/encode_data.pkl", "wb") as f:
            pickle.dump(combined_dict, f)

    return combined_dict


class SequenceDataset(Dataset):
    def __init__(
        self,
        seqs: np.ndarray,
        c: torch.Tensor,
        transform: T.Compose = T.Compose([T.ToTensor()]),
    ):
        "Initialization"
        self.seqs = seqs
        self.c = c
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.seqs)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        image = self.seqs[index]

        if self.transform:
            x = self.transform(image)
        else:
            x = image

        y = self.c[index]

        return x, y
