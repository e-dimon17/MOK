"""mok_core — shared library for the MoK-54B training subnet (Stage 2).

Layout mirrors production subnet libraries:
config schemas & the on-chain manifest, determinism primitives, the MoK model
stack, the deterministic data pipeline, and the chain/storage/telemetry clients.

SPEC_VERSION is the consensus protocol version. Any change to a golden-vector
constant (PRF outputs, canonical hashes, payload bytes, state roots) MUST bump
it — nodes on different spec versions refuse to interoperate.
"""

__version__ = "0.1.0"
SPEC_VERSION = 1
