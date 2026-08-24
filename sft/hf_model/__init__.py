"""Custom `trust_remote_code` HF model files for MoK-54B (model_type="mok_moe").

Import the classes explicitly (imports transformers):

    from sft.hf_model.configuration_mok_moe import MokMoeConfig
    from sft.hf_model.modeling_mok_moe import MokMoeForCausalLM

Both files are copied verbatim into converted model directories by
sft/convert_dcp_to_hf.py so downstream consumers can load with
`trust_remote_code=True` (and the rl vLLM plugin can reuse the config class).
"""
