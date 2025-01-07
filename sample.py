import argparse
import itertools

import torch

from dnadiffusion.data.dataloader import load_data
from dnadiffusion.metrics.metrics import generate_heatmap, kl_heatmap
from dnadiffusion.models.diffusion import Diffusion
from dnadiffusion.models.unet import UNet
from dnadiffusion.utils.sample_util import create_sample


def sample(model_path: str, num_samples: int = 1000, output_prefix: str = "final"):

    # Load checkpoint
    print("Loading checkpoint")
    checkpoint_dict = torch.load(model_path)
    length = checkpoint_dict["length"]
    right_aligned = checkpoint_dict["right_aligned"]
    n_tags = len(checkpoint_dict["tags"])

    print("Instantiating unet")
    unet = UNet(
        dim=length,
        channels=1,
        dim_mults=(1, 2, 4),
        resnet_block_groups=4,
        num_classes=(10,)*n_tags,
    )

    print("Instantiating diffusion class")
    diffusion = Diffusion(
        unet,
        timesteps=50,
    )

    # compatibility with older checkpoints
    state_dict = checkpoint_dict["model"]
    new_state_dict = {}
    for key, value in state_dict.items():
        if key == "model.label_emb.weight":
            new_key = "model.label_emb.0.weight"
        else:
            new_key = key
        new_state_dict[new_key] = value

    diffusion.load_state_dict(new_state_dict)

    # Send model to device
    print("Sending model to device")
    diffusion = diffusion.to("cuda")

    # Generating cell specific samples
    cell_num_lists = [ tag["cell_types"] for tag in checkpoint_dict["tags"] ]
    for group_numbers in itertools.product(*cell_num_lists):
        cond_name = [ checkpoint_dict["tags"][i]["numeric_to_tag"][group_numbers[i]] for i in range(len(group_numbers)) ]
        cond_name = "_".join(cond_name)
        print(f"Generating {num_samples} samples for labels {cond_name}")
        create_sample(
            diffusion,
            cond_name=cond_name,
            output_prefix=output_prefix,
            number_of_samples=num_samples // 10,
            group_number=group_numbers,
            cond_weight_to_metric=1,
            save_timesteps=False,
            save_dataframe=True,
            length=length,
            right_aligned=right_aligned
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dnadiffusion")
    parser.add_argument("--model", help="model file")
    parser.add_argument("--num-samples", type=int, default=1000, help="the number of samples")
    parser.add_argument("--output-prefix", default="final", help="output prefix for samples")
    args = parser.parse_args()

    sample(model_path=args.model, num_samples=args.num_samples, output_prefix=args.output_prefix)
