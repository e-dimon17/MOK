"""`mok-miner` console entry point (torchrun launches one process per rank)."""

from __future__ import annotations

import asyncio
import sys

from mok_core.telemetry import get_logger

log = get_logger("app.miner.main")


async def _amain(argv: list[str] | None) -> int:
    from .app import MinerApp  # noqa: PLC0415 — keep import side effects post-argparse
    from .bootstrap import bootstrap  # noqa: PLC0415

    ctx = await bootstrap("miner", argv)
    try:
        return await MinerApp(ctx).run()
    finally:
        await ctx.aclose()


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_amain(argv))
    except SystemExit as e:  # restart_required propagates its code
        return e.code if isinstance(e.code, int) else 1
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
