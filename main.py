import logging

import click
import uvicorn

import training


@click.group()
@click.option("--debug/--no-debug", default=False)
@click.pass_context
def main(ctx, debug: bool):
    # using context to pass the debug variable
    # into the sub-commands.
    ctx.ensure_object(dict)
    ctx.obj["DEBUG"] = debug

    format = "%(asctime)s [%(module)s] %(message)s"
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format=format,
        datefmt="[%X]",
    )


@main.group()
def train():
    """
    commands for running the model training.
    there is a sub-command for each model.
    """
    pass


@train.command()
def smp():
    """
    train an ARIMA model for predicting smp price.
    """
    training.smp(2)


@train.command()
def rf():
    """
    train an Random Forest model for considering effects of different parameters
    on the smp price.
    """
    training.rf(2)


@train.command()
def production():
    """
    train an ARIMA model for predicting smp production value.
    """
    training.production(2)


@main.command()
@click.pass_context
def serve(ctx):
    """
    serve model(s) using fastapi.
    """
    uvicorn.run(
        "server.server:app",
        host="0.0.0.0",
        port=8080,
        reload=ctx.obj["DEBUG"],
        workers=5,
        backlog=1024,
        log_level="debug" if ctx.obj["DEBUG"] else "info",
        use_colors=True,
        access_log=ctx.obj["DEBUG"],
    )


if __name__ == "__main__":
    main()
