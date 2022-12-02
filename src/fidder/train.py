from pathlib import Path

import typer
from pytorch_lightning import Trainer

from fidder._cli import OPTION_PROMPT_KWARGS, cli
from .data import FidderDataModule
from .model import Fidder


@cli.command(name="train", no_args_is_help=True)
def train_fidder(
    dataset_directory: Path = typer.Option(..., **OPTION_PROMPT_KWARGS),
    output_directory: Path = typer.Option("training", **OPTION_PROMPT_KWARGS),
    batch_size: int = 1,
    gradient_steps: int = 600,
    learning_rate: float = 1e-5,
) -> None:
    """Train a semantic segmentation model for fiducial detection."""
    model = Fidder(batch_size=batch_size, learning_rate=learning_rate)
    data_module = FidderDataModule(dataset_directory)

    trainer = Trainer(
        accelerator="auto",
        devices=1,
        auto_select_gpus=True,
        default_root_dir=output_directory,
        max_steps=gradient_steps,
        log_every_n_steps=10,
        check_val_every_n_epoch=None,
        val_check_interval=20,
        num_sanity_val_steps=0,
        enable_checkpointing=True,
    )
    trainer.fit(model, datamodule=data_module)
    return None
