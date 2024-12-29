import copy
import random
from typing import Any

import numpy as np
import torch
import torchvision.transforms as T
from accelerate import Accelerator
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from dnadiffusion.data.dataloader import SequenceDataset
from dnadiffusion.utils.utils import EMA


class TrainLoop:
    def __init__(
        self,
        data: dict[str, Any],
        model: torch.nn.Module,
        accelerator: Accelerator,
        epochs: int = 10000,
        log_step_show: int = 50,
        save_epoch: int = 500,
        #model_name: str = "model_48k_sequences_per_group_K562_hESCT0_HepG2_GM12878_12k",
        out_dir: str = "checkpoints",
        image_size: int = 200,
        right_aligned: bool = False,
        batch_size: int = 960,
    ):
        self.encode_data = data
        self.model = model
        self.optimizer = Adam(self.model.parameters(), lr=1e-4)
        self.accelerator = accelerator
        self.epochs = epochs
        self.log_step_show = log_step_show
        self.save_epoch = save_epoch
        self.out_dir = out_dir
        self.image_size = image_size
        self.right_aligned = right_aligned

        if self.accelerator.is_main_process:
            self.ema = EMA(0.995)
            self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)

        # Metrics
        self.train_kl, self.test_kl, self.shuffle_kl = 1, 1, 1

        self.start_epoch = 1

        # Dataloader
        seq_dataset = SequenceDataset(seqs=self.encode_data["X_train"], c=self.encode_data["x_train_cell_type"])
        self.train_dl = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    def train_loop(self):
        # Prepare for training
        self.model, self.optimizer, self.train_dl = self.accelerator.prepare(self.model, self.optimizer, self.train_dl)

        # Initialize wandb
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                "dnadiffusion",
                init_kwargs={"wandb": {"notes": "testing wandb accelerate script"}},
            )

        for epoch in tqdm(range(self.start_epoch, self.epochs + 1)):
            self.model.train()

            # Getting loss of current batch
            for step, batch in enumerate(self.train_dl):
                self.global_step = epoch * len(self.train_dl) + step

                loss = self.train_step(batch)

                # Logging loss
                if self.global_step % self.log_step_show == 0 and self.accelerator.is_main_process:
                    self.log_step(loss, epoch)

            # Saving model
            if epoch % self.save_epoch == 0 and self.accelerator.is_main_process:
                self.save_model(epoch)

    def train_step(self, batch):
        x, y = batch

        with self.accelerator.autocast():
            loss = self.model(x, y)

        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        self.accelerator.wait_for_everyone()
        self.optimizer.step()

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.ema.step_ema(self.ema_model, self.accelerator.unwrap_model(self.model))

        self.accelerator.wait_for_everyone()
        return loss

    def log_step(self, loss, epoch):
        if self.accelerator.is_main_process:
            self.accelerator.log(
                {
                    "loss": loss.mean().item(),
                    "epoch": epoch,
                },
                step=self.global_step,
            )


    def save_model(self, epoch):
        checkpoint_dict = {
            "model": self.accelerator.get_state_dict(self.model),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "ema_model": self.accelerator.get_state_dict(self.ema_model),
            "random": random.getstate(),
            "np_random": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_random": torch.random.get_rng_state(),
            "cuda_random": torch.cuda.get_rng_state(),
            "cuda_random_all": torch.cuda.get_rng_state_all(),
            "tags": self.encode_data["tags"],
            "length": self.image_size,
            "right_aligned": self.right_aligned,
        }
        torch.save(
            checkpoint_dict,
            f"{self.out_dir}/epoch_{epoch}.pt",
        )

    def load(self, path):
        checkpoint_dict = torch.load(path)
        self.model.load_state_dict(checkpoint_dict["model"])
        self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
        self.start_epoch = checkpoint_dict["epoch"]

        if self.accelerator.is_main_process:
            self.ema_model.load_state_dict(checkpoint_dict["ema_model"])

        random.setstate(checkpoint_dict["random"])
        np.random.set_state(checkpoint_dict["np_random"])
        torch.set_rng_state(checkpoint_dict["torch"])
        torch.random.set_rng_state(checkpoint_dict["torch_random"])
        torch.cuda.set_rng_state(checkpoint_dict["cuda_random"])
        torch.cuda.torch.cuda.set_rng_state_all(checkpoint_dict["cuda_random_all"])

        self.train_loop()
