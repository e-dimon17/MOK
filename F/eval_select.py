"""SFT checkpoint selection by an IFEval-style instruction-following battery.

Playbook: "Checkpoint selection by held-out IFEval/MT-Bench-style eval, not
train loss." This module scores every saved `checkpoint-*` directory with ~15
verifiable instruction probes — each judged by a pure regex/structure check on
the generated text — and picks the best scorer (ties -> later checkpoint).

The judges are pure functions (unit-tested on canned outputs); generation is
injectable so the selection logic itself never needs a GPU.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

GenerateFn = Callable[[Path, Sequence[str]], Sequence[str]]


@dataclass(frozen=True)
class Probe:
    name: str
    prompt: str
    judge: Callable[[str], bool]


# --------------------------------------------------------------------------- #
# Judges — pure text predicates
# --------------------------------------------------------------------------- #


def _judge_three_bullets(text: str) -> bool:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return len(lines) == 3 and all(ln.lstrip().startswith("- ") for ln in lines)


def _judge_json_keys_ab(text: str) -> bool:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        return False
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return False
    return isinstance(obj, dict) and set(obj.keys()) == {"a", "b"}


def _judge_single_word(text: str) -> bool:
    return len(text.strip().split()) == 1


def _judge_all_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def _judge_lowercase(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and all(c.islower() for c in letters)


def _judge_no_letter_e(text: str) -> bool:
    return bool(text.strip()) and "e" not in text.lower()


def _judge_ends_that_is_all(text: str) -> bool:
    return text.rstrip().endswith("That is all.")


def _judge_under_25_words(text: str) -> bool:
    words = text.split()
    return 0 < len(words) < 25


def _judge_ocean_thrice(text: str) -> bool:
    return len(re.findall(r"\bocean\b", text.lower())) == 3


def _judge_numbered_1_to_4(text: str) -> bool:
    nums = [int(m) for m in re.findall(r"^\s*(\d+)\.", text, re.MULTILINE)]
    return nums == [1, 2, 3, 4]


def _judge_starts_certainly(text: str) -> bool:
    return text.lstrip().startswith("Certainly:")


def _judge_no_commas(text: str) -> bool:
    return bool(text.strip()) and "," not in text


def _judge_two_paragraphs(text: str) -> bool:
    paragraphs = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    return len(paragraphs) == 2


def _judge_json_list_of_five(text: str) -> bool:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match is None:
        return False
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return False
    return isinstance(obj, list) and len(obj) == 5


def _judge_three_lines(text: str) -> bool:
    return len([ln for ln in text.strip().splitlines() if ln.strip()]) == 3


def _judge_yes_or_no(text: str) -> bool:
    return text.strip() in ("yes", "no")


PROBES: tuple[Probe, ...] = (
    Probe(
        "three_bullets",
        "List three benefits of exercise. Answer in exactly 3 bullet points, "
        "each starting with '- ', and nothing else.",
        _judge_three_bullets,
    ),
    Probe(
        "json_keys_ab",
        "Respond with a JSON object having exactly the keys \"a\" and \"b\" "
        "(values of your choice). Output only the JSON.",
        _judge_json_keys_ab,
    ),
    Probe(
        "single_word",
        "What is the capital of France? Answer with a single word only.",
        _judge_single_word,
    ),
    Probe(
        "all_caps",
        "Describe the sky in one sentence, in ALL CAPITAL LETTERS.",
        _judge_all_caps,
    ),
    Probe(
        "lowercase_only",
        "write one sentence about tea entirely in lowercase letters, no capitals anywhere.",
        _judge_lowercase,
    ),
    Probe(
        "no_letter_e",
        "Write one short sentence about cats without using the letter 'e' at all.",
        _judge_no_letter_e,
    ),
    Probe(
        "ends_that_is_all",
        "Give a one-sentence weather report and end your response with the exact "
        "phrase 'That is all.'",
        _judge_ends_that_is_all,
    ),
    Probe(
        "under_25_words",
        "Summarize the plot of Romeo and Juliet in fewer than 25 words.",
        _judge_under_25_words,
    ),
    Probe(
        "ocean_thrice",
        "Write a short paragraph that uses the word 'ocean' exactly three times.",
        _judge_ocean_thrice,
    ),
    Probe(
        "numbered_1_to_4",
        "Give four steps for making toast as a numbered list: lines starting "
        "'1.' through '4.', nothing else.",
        _judge_numbered_1_to_4,
    ),
    Probe(
        "starts_certainly",
        "Explain what a compiler does. Begin your response with exactly 'Certainly:'.",
        _judge_starts_certainly,
    ),
    Probe(
        "no_commas",
        "Describe your favorite meal in two sentences without using any commas.",
        _judge_no_commas,
    ),
    Probe(
        "two_paragraphs",
        "Write about mountains in exactly two paragraphs separated by a blank line.",
        _judge_two_paragraphs,
    ),
    Probe(
        "json_list_of_five",
        "Output a JSON array containing exactly five fruit names. Output only the JSON array.",
        _judge_json_list_of_five,
    ),
    Probe(
        "three_lines",
        "Write a haiku about autumn on exactly three lines.",
        _judge_three_lines,
    ),
    Probe(
        "yes_or_no",
        "Is 17 a prime number? Answer with exactly 'yes' or 'no' in lowercase.",
        _judge_yes_or_no,
    ),
)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def judge_response(probe: Probe, response: str) -> bool:
    return bool(probe.judge(response))


def score_outputs(responses: Mapping[str, str]) -> float:
    """Fraction of PROBES passed; a probe with no response fails."""
    passed = sum(1 for p in PROBES if p.name in responses and judge_response(p, responses[p.name]))
    return passed / len(PROBES)


def default_generate_fn(checkpoint: Path, prompts: Sequence[str]) -> list[str]:
    """Greedy ChatML-prompted generation with the checkpoint's own tokenizer.

    Heavy imports are local; requires the checkpoint dir to contain tokenizer
    files (F/convert_dcp_to_hf.py writes them, Trainer checkpoints inherit).
    """
    import torch  # noqa: PLC0415
    from transformers import PreTrainedTokenizerFast  # noqa: PLC0415

    from F.data_prep import IM_END, IM_START  # noqa: PLC0415
    from F.hf_model.modeling_mok_moe import MokMoeForCausalLM  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MokMoeForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16).to(device)
    model.eval()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint)
    outputs: list[str] = []
    for prompt in prompts:
        text = f"{IM_START}user\n{prompt}{IM_END}\n{IM_START}assistant\n"
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            generated = model.generate(
                input_ids, max_new_tokens=256, do_sample=False, tokenizer=tokenizer
            )
        completion = tokenizer.decode(generated[0, input_ids.shape[1] :], skip_special_tokens=True)
        outputs.append(completion.split(IM_END)[0].strip())
    return outputs


def score_checkpoint(checkpoint: Path, generate_fn: GenerateFn) -> float:
    responses = generate_fn(checkpoint, [p.prompt for p in PROBES])
    if len(responses) != len(PROBES):
        raise ValueError(
            f"generate_fn returned {len(responses)} responses for {len(PROBES)} probes"
        )
    return score_outputs({p.name: r for p, r in zip(PROBES, responses, strict=True)})


def list_checkpoints(run_dir: Path) -> list[Path]:
    """Trainer-style checkpoint-<step> subdirs (step order); falls back to
    `run_dir` itself when it is directly a model dir."""
    run_dir = Path(run_dir)
    found = [p for p in run_dir.glob("checkpoint-*") if p.is_dir() and (p / "config.json").is_file()]
    found.sort(key=lambda p: int(re.sub(r"\D", "", p.name) or 0))
    final = run_dir / "final"
    if final.is_dir() and (final / "config.json").is_file():
        found.append(final)
    if not found and (run_dir / "config.json").is_file():
        found = [run_dir]
    return found


def pick_checkpoint(
    run_dir: Path, *, generate_fn: GenerateFn | None = None
) -> tuple[Path, dict[str, float]]:
    """Score every checkpoint under `run_dir`; return (best, all scores).

    Ties go to the LATEST checkpoint (more training at equal instruction
    following). Raises FileNotFoundError when no checkpoint dir is found.
    """
    checkpoints = list_checkpoints(run_dir)
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint directories under {run_dir}")
    generate = generate_fn if generate_fn is not None else default_generate_fn
    scores = {str(ckpt): score_checkpoint(ckpt, generate) for ckpt in checkpoints}
    best = max(enumerate(checkpoints), key=lambda pair: (scores[str(pair[1])], pair[0]))[1]
    return best, scores


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mok-sft-select",
        description="Pick the best SFT checkpoint by the IFEval-style probe battery.",
    )
    parser.add_argument("run_dir", type=Path, help="SFT output dir containing checkpoint-*/")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    best, scores = pick_checkpoint(args.run_dir)
    for path, score in scores.items():
        marker = " <== best" if Path(path) == best else ""
        print(f"{score:6.3f}  {path}{marker}")
    print(f"selected: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
