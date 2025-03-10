#%%
from Bio import SeqIO
import pandas as pd
# %%
mfes = []
seqs = []
for record in SeqIO.parse("../../../FiveSpecies_Cao_allutr_with_energy_structure.fasta", "fasta"):
    mfe = float(record.id.split("|")[0])
    mfes.append(mfe)
    seqs.append(str(record.seq))
df = pd.DataFrame({"MFE": mfes, "seq": seqs})
# %%
df["len"] = df["seq"].apply(len)
df["AVG MFE"] = df["MFE"] / df["len"]
df["TAG"] = "middle"
am_mean = df["AVG MFE"].mean()
am_std = df["AVG MFE"].std()
df.loc[df["AVG MFE"]>am_mean+am_std, ["TAG"]] = "high"
df.loc[df["AVG MFE"]<am_mean-am_std, ["TAG"]] = "low"
#df.loc[df["AVG MFE"]>df["AVG MFE"].quantile(0.9), ["TAG"]] = "high"
#df.loc[df["AVG MFE"]<df["AVG MFE"].quantile(0.1), ["TAG"]] = "low"
# %%
from dnadiffusion.utils.utils import one_hot_encode_r
# %%
def one_hot_encode_r(seq, alphabet, max_seq_len):
    """One-hot encode a sequence."""
    seq_len = min(len(seq), max_seq_len)
    seq_array = np.zeros((max_seq_len, len(alphabet)))
    for i in range(-seq_len, 0):
        seq_array[i, alphabet.index(seq[i])] = 1
    return seq_array

nucleotides = ["A", "C", "G", "T"]

# %%
import numpy as np
one_hot_encode_r("ACGT", nucleotides, 200)
# %%
