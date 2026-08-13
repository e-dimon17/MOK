# MOK Subnet Operator Guide — Step by Step, A to H

This is the practical manual: what to run, in what order, on which machine,
and what you should see. It assumes no prior knowledge of the codebase.
Read the top-level `README.md` first for the one-page overview.

---

## 0. The big picture — who does what, and when

There are four roles. You (the project owner) play the first one; other people
(or you, during testing) play the rest.

| Role | Hardware | What they do |
|---|---|---|
| **Owner** | any CPU machine + occasional GPU rental | prepares the dataset (A), publishes the run manifest and initial model (B), runs post-training and releases (F–H) |
| **Miner** | 8× B300 (SM103) NVLink node | trains: runs windows, uploads compressed gradients (C–E) |
| **Scoring validator** | 1× H200/B200-class GPU (≥141 GB) | scores miners' gradients by loss improvement, sets chain weights (C) |
| **Audit validator** | 8× B300 node (identical to miners) | replays sampled miner windows bitwise; a hash mismatch = slash (C) |

The lifecycle in one line:

```
A (freeze data) → B (bring up fleet) → C (bulk pretraining, weeks)
→ D (quality anneal, days) → E (16k context, ~1 day)
→ F (SFT) → G (DPO + RLVR) → H (evaluate + release)      ← owner side, per release
```

### Server requirements at a glance

| Step | Machine class | Count |
|---|---|---|
| A | big CPU box (no GPU) | 1 |
| B | 1× Tier-A GPU node (owner) + each miner's own Tier-A node | 1 + fleet |
| C–E | miners: Tier-A · scoring validators: 1× ≥141 GB GPU · auditors: Tier-A | 15–25 + 2–3 + 3 |
| F | 1–2× 8-GPU nodes (H200-class or better — B300 not required) | 1–2 |
| G | 1 training node + 1–3 vLLM rollout nodes | 2–4 |
| H | 1× node with ≥141 GB GPU (evals); any CPU (provenance) | 1 |

**"Tier-A node"** (the standardized miner/auditor unit, enforced by
`mok-onboard`'s preflight): **8× NVIDIA B300 (SM103)** in one NVLink domain,
288 GB HBM each, **≥1.2 TB system RAM**, 64+ CPU cores, **≥3 TB free NVMe**
for shards + checkpoints, ≥1 Gbps symmetric internet, Linux + the blessed
container. Own (~$300–500k) or rent (~$25–50k/month cloud).

Steps C, D, E are the *same program* — D and E only change the on-chain
manifest phase. Steps A, F, G, H are owner-side tools, not subnet protocol.

---

## 1. One-time setup (every machine, every role)

### 1.1 Accounts and credentials you need

- **Cloudflare R2** (or any S3-compatible storage): one bucket per participant.
  Create the bucket, a **write** key pair (kept private) and a **read** key
  pair (committed on-chain so peers can fetch from you).
- **Bittensor wallet**: install `btcli`, create a coldkey + hotkey
  (`btcli wallet create`). You need a small amount of TAO to register.
- Optional: **HuggingFace token** (step A downloads, step H uploads),
  **Weights & Biases key** (dashboards).

### 1.2 Install

```bash
cd /workspace/MOK
cp .env.example .env        # fill in R2_*, BT_*, HF_TOKEN — see comments in the file
python3.12 -m venv .venv && source .venv/bin/activate

# CPU machine (owner tools, step A, all CPU tests):
pip install -e ".[data,dev]"

# GPU node (miner / auditor — SM103 only; the MoK kernel refuses other GPUs):
pip install -e ".[gpu,dev]" --no-build-isolation

# Post-training machine (steps F/G/H):
pip install -e ".[post,dev]"
```

Sanity check on any machine (takes ~7 min, needs no GPU and no network):

```bash
pytest -q          # expect: 1110 passed
```

### 1.3 The golden rule

Everything consensus-critical (data, configs, container image, seeds) is
**frozen and hashed** before the run starts. If a command asks for a hash or a
manifest, it is protecting you from silently diverging from the fleet. Never
edit `C/configs/base.yaml` after the run has started — changes go through
manifest amendments (see step D).

---

## 2. Step A — Prepare the dataset (owner · CPU machine · ~days)

**Server requirements (no GPU at all):**

| | Minimum | Recommended |
|---|---|---|
| CPU | 32 cores | 128 cores (tokenize/dedup are parallel) |
| RAM | 128 GB | 512 GB (MinHash index for dedup) |
| Disk | 12 TB NVMe | 30 TB NVMe (raw spools + tokens + shards coexist) |
| Network | 1 Gbps | 10 Gbps (corpus download is ~10+ TB) |

A rented bare-metal CPU box does this for ~$1–2k total. The `--smoke` test
needs only a laptop.

**What it does:** turns public text corpora into frozen, verifiable training
data: tokenizer → tokenized 4096-token sequences → 512 MB content-addressed
shards → a Merkle root that goes on-chain. After this step, nobody (including
you) can change a single byte without every node noticing.

**Start with the smoke test** (5 minutes, no downloads — proves the whole
pipeline on your machine):

```bash
mok-data --smoke tests/fixtures/tiny_corpus.txt --out mok-data-smoke
# --smoke takes one or more local text files as the miniature corpus
# expect: prints a Merkle root; re-running prints the SAME root (determinism)
```

**The real run**, command by command (disk needed: ~10 TB scratch for the
bulk corpus; each stage is resumable):

```bash
SPOOL=/data/spool; TOK=/data/tokens; SHARDS=/data/shards

# 1. Download the sources listed in the corpus config (FineWeb-Edu, DCLM, code, math)
mok-data download --corpus-config A/configs/corpus_bulk.yaml --spool-dir $SPOOL

# 2. Cross-source dedup
mok-data dedup --corpus-config A/configs/corpus_bulk.yaml --spool-dir $SPOOL --out-dir $SPOOL-dedup

# 3. Train the frozen 65k tokenizer (once, ever — it is part of the consensus)
mok-data tokenizer --corpus-config A/configs/corpus_bulk.yaml \
  --spool-dir $SPOOL-dedup --out tokenizer.json

# 4. Tokenize + 5. pack into shards + 6. build the Merkle manifest
mok-data tokenize --corpus-config A/configs/corpus_bulk.yaml \
  --spool-dir $SPOOL-dedup --tokenizer tokenizer.json --out-dir $TOK
mok-data shard   --corpus-config A/configs/corpus_bulk.yaml --tokens-dir $TOK --out-dir $SHARDS
mok-data manifest --corpus-config A/configs/corpus_bulk.yaml --out-dir $SHARDS --tokenizer tokenizer.json

# 7. Upload to your R2 bucket (resumable; uses R2_* env vars)
mok-data upload --data-dir $SHARDS

# 8. Verify (do this; it is what miners will do too)
mok-data verify --data-dir $SHARDS
```

Repeat with `A/configs/corpus_anneal.yaml` for the step-D anneal tree.
**Outputs to keep safe:** `tokenizer.json`, `manifest.json`, `shard_index.json`
— their hashes go into the run manifest in step B.

---

## 3. Step B — Bring up the fleet (owner + miners · first GPU money)

**Server requirements:**

| Task | Machine |
|---|---|
| Container build (3.1) | any Docker host with an NVIDIA toolchain, ~64 GB RAM, ~200 GB disk |
| Init publish (3.2) + calibration (3.4) | **1× Tier-A node** (rented is fine — this is your first B300 rental; `--backend reference` on CPU works for testnet rehearsal only) |
| Miner onboarding (3.3) | each miner's own **Tier-A node** — the preflight hard-enforces: 8× B300 (SM103), NVLink all-pairs, ≥280 GB VRAM/GPU, ≥1.2 TB RAM, ≥3 TB free NVMe, container digest present |
| Attestation verify (3.3) | verifier needs a Tier-A node once to precompute `derive_expected`; judging afterwards is CPU-only |

**What it does:** builds the one blessed container, publishes the initial
model, verifies each miner's hardware, and calibrates the kernel settings.

### 3.1 Owner: build the container

```bash
docker build -t mok-subnet:stage2 -f B/container/Dockerfile .
docker inspect --format='{{index .RepoDigests 0}}' mok-subnet:stage2
# record the sha256 digest — it goes in the manifest; every node must run this exact image
```

### 3.2 Owner: publish the initial model

On one GPU node (or CPU with `--backend reference` for testnet rehearsal):

```bash
mok-init-publish --config C/configs/base.yaml --local-dir checkpoints \
  --seed 42 --backend mok --device cuda
# prints the init state_root — this hash goes on-chain; every miner verifies against it
```

Then create the run manifest (run seed, dataset Merkle roots, config hash,
container digest, start block) and commit its hash on-chain. The manifest JSON
is built with `mok_core.config.build_manifest` — a worked example is in
`B/onboarding/init_publish.py`'s docstring.

### 3.3 Miner: onboard (each miner runs this once)

```bash
mok-onboard --config C/configs/base.yaml
# runs in order: hardware preflight (8×B300, NVLink, RAM, disk)
# → wallet check → subnet registration → R2 bucket credential commit
# → download + verify the init checkpoint → self-attestation
# use --skip-<step> flags to redo a single part
```

The **attestation** it ends with is the hardware proof: a 20-step deterministic
toy training run derived from a chain block hash. Only a real 8×SM103 NVLink
node produces the correct hash inside the deadline. Manual flow, if you need it:

```bash
mok-attest challenge --from-chain --out challenge.json            # anyone
mok-attest respond --challenge challenge.json --out response.json # miner, on the node
mok-attest verify --challenge challenge.json --response response.json \
  --expected-root <root> --issued-ts <t0> --received-ts <t1>      # validator
```

### 3.4 Owner: calibrate (on one rented node, before mainnet)

```bash
mok-calibrate rehearse --config C/configs/base.yaml ...   # loopback windows + determinism check
mok-calibrate sweep --config C/configs/base.yaml ...      # writes C/configs/mok_tuned.yaml
mok-calibrate adam-ab --config C/configs/base.yaml ...    # pins the Adam-reset policy
```

**Gate:** before real money, run the GPU test suite on the node —
`torchrun --standalone --nproc-per-node=8 -m pytest tests/gpu -m gpu -q`.
`test_03` (window determinism) and `test_06` (self-replay) **must pass**;
the README's Testing section explains the two-node replay gate.

---

## 4. Step C — The training run (everyone · weeks)

**Server requirements (per participant, for the whole run):**

| Role | Spec |
|---|---|
| Miner | **Tier-A node** (8× B300, 288 GB HBM each ⇒ ~208 GB/GPU used; ≥1.2 TB RAM; ≥3 TB NVMe holds assigned shards ≈ 300–500 GB + checkpoint history; ≥1 Gbps — payloads are ~1–2 GB per 45-min window up, ~20× that down) |
| Scoring validator | **1× GPU with ≥141 GB** (H200/B200/B300 — runs the 54B reference model forward-only, ~108 GB weights), 32+ cores, 256 GB RAM, 1 TB NVMe, 1 Gbps |
| Audit validator | **Tier-A node, identical to miners** — bitwise replay demands hardware identity; ≥3 recommended (2-of-3 slash quorum) |
| Leader validator (highest-stake scoring validator) | + ~2 TB NVMe extra (checkpoint uploads + aggregator objects) |

**Do not mix GPU models within a role.** An auditor on different silicon than
the miners will produce honest mismatches and slash innocent people.

Each role starts its long-running process (inside the container; the
`C/scripts/*.sh` wrappers do the `torchrun` incantation for you):

```bash
# Miner (8 GPUs):
bash C/scripts/run_miner.sh                 # = torchrun -n 8 -m C.miner.main --config ... --overlay ...

# Scoring validator (1 GPU):
bash C/scripts/run_validator.sh

# Audit validator (8 GPUs):
bash C/scripts/run_auditor.sh
```

Useful flags (all roles): `--network test|finney`, `--netuid N`,
`--state-dir /data/mok-state`, `--device`, and `--local-harness` (offline
loopback mode for rehearsals — no chain, no R2).

**What normal operation looks like** (one window ≈ 45 minutes):
train 500 inner steps → extract + compress the pseudo-gradient (~1–2 GB) →
commit its hash on-chain → upload → the leader validator publishes the window
certificate → every node downloads the certified peer set and applies the
identical outer step → checkpoint every 10 windows. Watch the JSONL logs in
`--state-dir` (or W&B): `loss`, `capacity_util` (must stay < 0.4 before the
capacity amendment), `router_entropy`, `audit pass-rate`.

**Things that self-heal:** a killed miner catches up from the latest
checkpoint by replaying certified windows (bitwise-verified); a desynced node
does the same automatically. **Things that page you:** repeated audit
mismatches on your own node (environment drift — check container digest,
driver, `docs/ENGINEERING_NOTES.md`), or a loss spike triggering a rollback
vote (validators handle it; miners just follow the manifest).

---

## 5. Step D — Quality anneal (subnet · ~4 days)

**Server requirements:** identical to step C — same fleet, same roles, no
change (that is the point of the phase-amendment design). Miners additionally
download the anneal shard tree (~50–100 GB per miner) before the boundary.

Still pretraining — only the data tree and LR schedule change, via an on-chain
**manifest amendment** (never a local config edit). To fork an interim release
(e.g., v0.9 at ~1.5T tokens) while the main branch keeps training:

```bash
python -m D.release_fork --checkpoint checkpoints/w00003000 \
  --manifest manifest.json --config C/configs/base.yaml \
  --decay-tokens 150000000000 --effective-window 3010 \
  --committed-block <block> --out release-v0.9/
# writes the forked manifest + RELEASE_FORK.md (the operator runbook for the fork)
```

The end-of-run anneal is the same amendment with `--decay-tokens 400000000000`
and the anneal dataset. Miners need no action — the phase table tells their
running processes what to do. Local rehearsal: `bash D/scripts/run_miner.sh`.

## 6. Step E — Context extension to 16k (subnet · ~1 day)

**Server requirements:** identical to step C. The 16k workspace adds ~1–2 GB
HBM per GPU (new MoK buffers) and attention activations grow ~4× per
sequence — both fit inside the Tier-A node's existing ~80 GB/GPU headroom.

Another phase amendment (`E/configs/context16k.yaml` holds the values:
seq 16384, RoPE θ=500k, new workspace shape). This one sets
`requires_restart: true` — miner processes exit cleanly at the boundary and
their supervisor (docker compose restart policy) relaunches them; the new MoK
workspace materializes automatically. Rehearsal: `bash E/scripts/run_miner.sh`.

---

## 7. Step F — SFT (owner · 1–2 GPU nodes · ~1 week)

**Server requirements:** post-training runs on **standard HF kernels — B300 is
NOT required** (any modern 8-GPU node works). Full fine-tuning of 54B needs
weights + grads + Adam ≈ 900 GB across the FSDP group:

| | Spec |
|---|---|
| Training | 1–2 nodes × 8 GPUs, **≥1.1 TB aggregate HBM** (8× H200 141 GB works; 2× 8× A100/H100-80GB nodes also work), 1 TB RAM, 2 TB NVMe |
| Conversion + parity gate (steps 1–2) | CPU-only, 256 GB RAM (loads the checkpoint on CPU), ~250 GB disk for the HF export |

```bash
# 1. Convert the annealed checkpoint to a HuggingFace model
python -m F.convert_dcp_to_hf checkpoints/w00005000 hf-base/ --tokenizer tokenizer.json

# 2. THE PARITY GATE — never skip this
python -m F.verify_conversion checkpoints/w00005000 hf-base/
# expect: max logit diff < 2e-2 and >99% argmax agreement; failure = stop and debug

# 3. Train (config: F/configs/sft.yaml — datasets, 2 epochs, seq 16k)
mok-sft --config F/configs/sft.yaml

# 4. Pick the best checkpoint by instruction-following probes
python -m F.eval_select <output-dir>
```

## 8. Step G — DPO + RLVR (owner · 2–4 GPU nodes · 1–2 weeks)

**Server requirements:**

| | Spec |
|---|---|
| DPO training | same as step F training (policy + frozen reference model ⇒ prefer the 2-node option or ZeRO-3 offload on 1 node) |
| GRPO training | 1× step-F-class node |
| vLLM rollout servers | 1–3 nodes, each **1–2× ≥141 GB GPUs** (the model serves at 5.5B-active cost; ~108 GB bf16 weights per replica) |
| Code-reward sandbox | CPU on the training node (bubblewrap/nsjail if available; rlimit fallback otherwise) — 32+ spare cores recommended |

```bash
# 1. DPO (cheap alignment gain; config: G/configs/dpo.yaml)
python -m G.dpo_train --config G/configs/dpo.yaml

# 2. RLVR — start vLLM rollout servers on 1–3 nodes, then:
mok-rl --config G/configs/grpo.yaml
# rewards are execution-checked: math answers via symbolic equivalence,
# code via sandboxed unit tests — this is where GSM8K/HumanEval jump
```

## 9. Step H — Evaluate + release (owner · 1 node · ~1 week)

**Server requirements:**

| | Spec |
|---|---|
| Benchmarks (`mok-eval`, vLLM backend) | 1 node, **1–2× ≥141 GB GPUs**, 128 GB RAM |
| Provenance bundle + `verify_bundle` | **any CPU machine** — deliberately: anyone must be able to verify a release without GPUs |
| `replay_window` (full bitwise re-execution) | **Tier-A node** — replay is the real training computation |
| HF upload | any machine; ~250 GB disk + good uplink for the weight shards |

```bash
# 1. Benchmarks
mok-eval --model-path hf-chat/ --backend vllm --out evals.json --markdown-out evals.md

# 2. Build the provenance bundle — the artifact no other model release has
python - <<'PY'
from H.provenance import build_bundle   # see its docstring for the full call
PY

# 3. Verify the bundle offline (anyone can do this — that's the point)
python -m H.verify_bundle release-bundle/            # expect: ok

# 4. Anyone can replay any window from the bundle:
python -m H.replay_window --bundle release-bundle/ --window 1234 --miner-uid 7 \
  --theta-start <checkpoint-dir>                      # exit 0 iff bitwise match

# 5. Publish
python -m H.hf_upload   # dry-run mode first; uploads weights + model card + bundle
```

---

## 10. Quick reference

| I want to… | Run |
|---|---|
| prove the data pipeline works, right now, on my laptop | `mok-data --smoke` |
| run the whole CPU test suite | `pytest -q` |
| check a GPU node is fit for duty | `torchrun --standalone --nproc-per-node=8 -m pytest tests/gpu -m gpu -q` |
| rehearse the full protocol offline (no chain, no R2) | `mok-miner --local-harness --uid 0 ...` |
| join as a miner | `mok-onboard`, then `bash C/scripts/run_miner.sh` |
| see why a miner scored zero | validator logs + `docs/INTERNAL_API.md` (scoring/slashing sections) |
| verify a released model's training history | `python -m H.verify_bundle <bundle>` then `H.replay_window` |

**Where to look when something breaks:** `docs/ENGINEERING_NOTES.md` (100+
known caveats, environment pins, GPU-milestone flags) → the module docstring
of whatever failed (every module documents its contract) → the matching test
file in `tests/unit/` (executable examples of correct usage).
