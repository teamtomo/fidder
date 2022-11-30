from pathlib import Path

import typer
from pytorch_lightning import Trainer

from .._cli import DEFAULT_OPTION_KWARGS, cli
from ..data import FidderDataModule
from .model import Fidder


@cli.command(name="train")
def train_fidder(
    dataset_directory: Path = typer.Option(..., **DEFAULT_OPTION_KWARGS),
    output_directory: Path = typer.Option("training", **DEFAULT_OPTION_KWARGS),
    gradient_steps: int = typer.Option(600, **DEFAULT_OPTION_KWARGS),
    batch_size: int = typer.Option(1, **DEFAULT_OPTION_KWARGS),
    learning_rate: float = typer.Option(1e-5, **DEFAULT_OPTION_KWARGS),
) -> None:
    """Train Fidder."""
    model = Fidder()
    data_module = FidderDataModule(dataset_directory)
    trainer = Trainer(
        accelerator="auto",
        default_root_dir=output_directory,
        max_steps=gradient_steps,
        check_val_every_n_epoch=None,
        val_check_interval=20,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=data_module)
    return None
