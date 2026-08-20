"""`mok-validator` console entry point."""

from __future__ import annotations

import asyncio
import os
import sys

from mok_core.telemetry import get_logger

log = get_logger("app.validator.main")


async def _amain(argv: list[str] | None) -> int:
    from C.miner.bootstrap import bootstrap  # noqa: PLC0415 — post-argparse imports

    from .app import ValidatorApp  # noqa: PLC0415

    ctx = await bootstrap("validator", argv)
    try:
        return await ValidatorApp(ctx).run()
    finally:
        await ctx.aclose()


def main(argv: list[str] | None = None) -> int:
    try:
        code = asyncio.run(_amain(argv))
    except SystemExit as e:  # restart/rollback codes propagate to the supervisor
        code = e.code if isinstance(e.code, int) else 1
    except KeyboardInterrupt:
        code = 0
    # Exit WITHOUT interpreter teardown: the substrate SDK's websocket destructor
    # can join a wedged keepalive thread forever, turning every exit path
    # (rollback restart, phase restart, SIGTERM) into a hang. All state is
    # persisted by the app before it raises; nothing below needs destructors.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    sys.exit(main())
