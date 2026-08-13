"""`mok-auditor` console entry point."""

from __future__ import annotations

import asyncio
import sys

from mok_core.telemetry import get_logger

log = get_logger("app.auditor.main")


async def _amain(argv: list[str] | None) -> int:
    from C.miner.bootstrap import bootstrap  # noqa: PLC0415 — post-argparse imports

    from .app import AuditorApp  # noqa: PLC0415

    ctx = await bootstrap("auditor", argv)
    try:
        return await AuditorApp(ctx).run()
    finally:
        await ctx.aclose()


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_amain(argv))
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
