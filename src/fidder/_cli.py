import typer

cli = typer.Typer(add_completion=False, no_args_is_help=True)
OPTION_PROMPT_KWARGS = {"prompt": True, "prompt_required": True}
