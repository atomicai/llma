from typer import Typer
from llma.api import index, generate

cli = Typer()

cli.command()(index.index)
cli.command()(generate.generate)

if __name__ == "__main__":
    cli()
