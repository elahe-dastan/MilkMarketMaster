import logging

import click
import uvicorn


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


@main.command()
@click.pass_context
def serve(ctx):
    uvicorn.run(
        "server.server:app",
        host="0.0.0.0",
        port=8080,
        reload=True if ctx.obj["DEBUG"] else False,
        workers=1,
    )


if __name__ == "__main__":
    main()
