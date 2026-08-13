"""Step F — supervised fine-tuning (centralized, TRL) on standard HF kernels.

Modules import heavy post-training deps (transformers/trl/datasets) lazily;
only `F.hf_model.{configuration,modeling}_mok_moe` import transformers at
module level, by design — they ARE the transformers integration.
"""
