import argparse

import torch

from dnadiffusion.data.dataloader import load_data
from dnadiffusion.metrics.metrics import generate_heatmap, kl_heatmap
from dnadiffusion.models.diffusion import Diffusion
from dnadiffusion.models.unet import UNet
from dnadiffusion.utils.sample_util import create_sample


def sample(model_path: str, num_samples: int = 1000, length: int = 200):

    # Ensure length is divisible by 8 for UNet
    length = length if length % 8 == 0 else (length // 8 + 1) * 8

    print("Instantiating unet")
    unet = UNet(
        dim=length,
        channels=1,
        dim_mults=(1, 2, 4),
        resnet_block_groups=4,
    )

    print("Instantiating diffusion class")
    diffusion = Diffusion(
        unet,
        timesteps=50,
    )

    # Load checkpoint
    print("Loading checkpoint")
    checkpoint_dict = torch.load(model_path)
    diffusion.load_state_dict(checkpoint_dict["model"])

    # Send model to device
    print("Sending model to device")
    diffusion = diffusion.to("cuda")

    # Generating cell specific samples
    cell_num_list = checkpoint_dict["tag"]["cell_types"]
    numeric_to_tag = checkpoint_dict["tag"]["numeric_to_tag"]

    for i in cell_num_list:
        print(f"Generating {num_samples} samples for label {numeric_to_tag[i]}")
        create_sample(
            diffusion,
            conditional_numeric_to_tag=numeric_to_tag,
            cell_types=cell_num_list,
            number_of_samples=int(num_samples / 10),
            group_number=i,
            cond_weight_to_metric=1,
            save_timesteps=False,
            save_dataframe=True,
            length=length,
            right_aligned=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dnadiffusion")
    parser.add_argument("--model", help="model file")
    parser.add_argument("--num-samples", type=int, default=1000, help="the number of samples")
    parser.add_argument("--length", type=int, default=200, help="sequence length")
    args = parser.parse_args()

    sample(model_path=args.model, num_samples=args.num_samples, length=args.length)
