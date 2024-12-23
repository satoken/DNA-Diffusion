import argparse

from accelerate import Accelerator

from dnadiffusion.data.dataloader import load_data_rnaseq
from dnadiffusion.models.diffusion import Diffusion
from dnadiffusion.models.unet import UNet
from dnadiffusion.utils.train_util import TrainLoop


def train(batch_size: int, epochs: int, save_epoch: int, length: int, model_name: str):
    accelerator = Accelerator(split_batches=True, log_with=["wandb"])
    accelerator.init_trackers(project_name="dnadiffusion", init_kwargs={"wandb": {"mode": "offline"}})

    data = load_data_rnaseq(
        data_path="./data/HEK_sequence.csv",
        max_seq_len=length,
        #num_sampling_to_compare_cells=1000,
    )

    unet = UNet(
        dim=length,
        channels=1,
        dim_mults=(1, 2, 4),
        resnet_block_groups=4,
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
        sample_epoch=500,
        save_epoch=save_epoch,
        model_name=model_name,
        image_size=length,
        num_sampling_to_compare_cells=1000,
        batch_size=batch_size,
    ).train_loop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dnadiffusion")
    parser.add_argument("--batch-size", type=int, default=480, help="batch size")
    parser.add_argument("--epochs", type=int, default=10000, help="the number of epochs")
    parser.add_argument("--save-epoch", type=int, default=500, help="the interval of saving the model")
    parser.add_argument("--length", type=int, default=200, help="sequence length")
    parser.add_argument("--model-name", type=str, default="5utr_rnaseq", help="model name")
    args = parser.parse_args()

    train(batch_size=args.batch_size, epochs=args.epochs, save_epoch=args.save_epoch, length=args.length, model_name=args.model_name)
