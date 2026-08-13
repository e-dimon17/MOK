"""Step G — DPO + RLVR post-training (centralized, TRL + vLLM rollouts).

Layout:
  - `G.dpo_train` / `G.grpo_train`: TRL harnesses (`mok-rl` = grpo_train);
  - `G.rewards`: verifiable-task reward functions (sympy math, sandboxed code);
  - `G.data`: RLVR prompt-set builders (GSM8K/MATH, MBPP + self-generated);
  - `G.vllm_plugin`: out-of-tree vLLM registration of the MoK architecture.

Heavy post-training deps (transformers/trl/datasets/vllm/sympy) import lazily
inside functions; every module here is importable on any host.
"""
