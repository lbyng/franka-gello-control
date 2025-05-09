# ruff: noqa
import os
from . import algs, datasets, envs, networks, processors

__all__ = ["algs", "datasets", "envs", "networks", "processors"]

if len(os.environ["RESEARCH_LIGHTNING_MATPLOTLIB_MODE"]) > 0:
    import matplotlib as mpl

    mpl.use(os.environ["RESEARCH_LIGHTNING_MATPLOTLIB_MODE"])

if os.environ["RESEARCH_LIGHTNING_RICH_PRINT"] == "true":
    from rich.traceback import install
    from rich import print as rprint
    import builtins

    def print(*args, **kwargs):  # ruff: noqa
        if len(args) > 0 and "gamify" in repr(args[0]):
            pid = os.getpid()
            prefix = f"[bold blue]gamify[/bold blue] {pid}"
            rprint(prefix, *args, **kwargs)
        else:
            rprint(*args, **kwargs)

    builtins.print = print

    install(show_locals=os.environ.get("LOCALS", "true") == "true")
