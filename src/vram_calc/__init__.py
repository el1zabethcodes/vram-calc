"""vram-calc -- precise VRAM calculator for quantized GGUF models.

Quick start::

    >>> from vram_calc import estimate_vram, plan_offload, resolve_gpu
    >>> est = estimate_vram("qwen2.5-32b", quant="Q4_K_M", context_len="32k")
    >>> plan = plan_offload(est, resolve_gpu("rtx-4090-24g"))
    >>> plan.status.value, plan.ngl_flag
    ('partial', 54)
"""

from .calculator import (
    MODELS,
    Backend,
    CacheQuant,
    Estimate,
    ModelPreset,
    Quant,
    attention_workspace,
    custom_model,
    estimate_kv_cache,
    estimate_vram,
    estimate_weights,
    parse_context,
    resolve_backend,
    resolve_cache_quant,
    resolve_model,
    resolve_quant,
)
from .hardware import (
    GPUS,
    FitStatus,
    GPU,
    OffloadPlan,
    custom_gpu,
    plan_all,
    plan_offload,
    resolve_gpu,
    usable_mb,
)

__version__ = "0.1.0"

__all__ = [
    "MODELS",
    "GPUS",
    "Backend",
    "CacheQuant",
    "Estimate",
    "FitStatus",
    "GPU",
    "ModelPreset",
    "OffloadPlan",
    "Quant",
    "attention_workspace",
    "custom_gpu",
    "custom_model",
    "estimate_kv_cache",
    "estimate_vram",
    "estimate_weights",
    "parse_context",
    "plan_all",
    "plan_offload",
    "resolve_backend",
    "resolve_cache_quant",
    "resolve_gpu",
    "resolve_model",
    "resolve_quant",
    "usable_mb",
    "__version__",
]
