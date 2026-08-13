"""Out-of-tree vLLM registration of the MoK-MoE architecture (step G).

    from G.vllm_plugin import register_mok_moe
    register_mok_moe()   # then vLLM can serve converted MokMoe checkpoints

vllm itself is imported lazily and guarded; the name-mapping tables in
`G.vllm_plugin.mok_moe_vllm` are pure and importable everywhere.
"""

from G.vllm_plugin.mok_moe_vllm import (
    HF_ARCHITECTURE,
    VLLM_CLASS_PATH,
    expert_params_mapping,
    map_dense_name,
    register_mok_moe,
)

__all__ = [
    "HF_ARCHITECTURE",
    "VLLM_CLASS_PATH",
    "expert_params_mapping",
    "map_dense_name",
    "register_mok_moe",
]
