"""Core VRAM math for quantized GGUF models.

All internal sizes are computed in binary units (GiB) and displayed as "GB"
-- the convention used across the local-LLM community (a "24 GB" RTX 4090
is 24576 MiB). Every approximation is documented at its use site and
calibrated against real llama.cpp / GGUF behavior; expect ~±5% accuracy
against actual `nvidia-smi` readings, not bit-exact accounting.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple, Union

GiB = 1024**3
MiB = 1024**2


def to_gib(num_bytes: float) -> float:
    return num_bytes / GiB


# ---------------------------------------------------------------------------
# Quantization formats
# ---------------------------------------------------------------------------


class Quant(str, Enum):
    """GGUF quantization formats with their *effective* bits-per-weight.

    Effective bpw = (file size x 8) / parameter count, measured on real
    GGUF files. It is higher than the raw tensor bpw for mid/low k-quants
    because `tok_embd` / `output.weight` are kept at Q6_K/f16 precision,
    and roughly equal for I-quants and Q8_0 (which quantize everything).
    Values are calibrated against published file sizes, ±3%.
    """

    IQ1_S = "IQ1_S"
    IQ1_M = "IQ1_M"
    IQ2_XXS = "IQ2_XXS"
    IQ2_XS = "IQ2_XS"
    IQ2_S = "IQ2_S"
    IQ2_M = "IQ2_M"
    IQ3_XXS = "IQ3_XXS"
    IQ3_XS = "IQ3_XS"
    IQ3_S = "IQ3_S"
    IQ3_M = "IQ3_M"
    Q2_K_S = "Q2_K_S"
    Q2_K = "Q2_K"
    Q3_K_S = "Q3_K_S"
    Q3_K_M = "Q3_K_M"
    Q3_K_L = "Q3_K_L"
    IQ4_XS = "IQ4_XS"
    IQ4_NL = "IQ4_NL"
    Q4_0 = "Q4_0"
    Q4_K_S = "Q4_K_S"
    Q4_K_M = "Q4_K_M"
    Q4_1 = "Q4_1"
    Q5_0 = "Q5_0"
    Q5_K_S = "Q5_K_S"
    Q5_K_M = "Q5_K_M"
    Q5_1 = "Q5_1"
    Q6_K = "Q6_K"
    Q8_0 = "Q8_0"
    BF16 = "BF16"
    F16 = "F16"

    @property
    def bpw(self) -> float:
        return _EFFECTIVE_BPW[self]

    @property
    def head_bpw(self) -> float:
        """Assumed precision of the embedding/output tensors for split planning.

        llama.cpp keeps these tensors at higher precision than the body:
        never below ~Q4_0 (very-low-bit I-quants still protect them) and
        never above f16. Calibrated against real per-layer offload numbers.
        """
        return min(16.0, max(4.5, self.bpw))


_EFFECTIVE_BPW: Dict[Quant, float] = {
    Quant.IQ1_S: 1.56,
    Quant.IQ1_M: 1.75,
    Quant.IQ2_XXS: 2.06,
    Quant.IQ2_XS: 2.31,
    Quant.IQ2_S: 2.50,
    Quant.IQ2_M: 2.70,
    Quant.IQ3_XXS: 3.06,
    Quant.IQ3_XS: 3.30,
    Quant.IQ3_S: 3.44,
    Quant.IQ3_M: 3.66,
    Quant.Q2_K_S: 3.20,
    Quant.Q2_K: 3.35,
    Quant.Q3_K_S: 3.50,
    Quant.Q3_K_M: 3.91,
    Quant.Q3_K_L: 4.27,
    Quant.IQ4_XS: 4.25,
    Quant.IQ4_NL: 4.50,
    Quant.Q4_0: 4.55,
    Quant.Q4_K_S: 4.58,
    Quant.Q4_K_M: 4.85,
    Quant.Q4_1: 5.10,
    Quant.Q5_0: 5.54,
    Quant.Q5_K_S: 5.52,
    Quant.Q5_K_M: 5.69,
    Quant.Q5_1: 6.00,
    Quant.Q6_K: 6.59,
    Quant.Q8_0: 8.50,
    Quant.BF16: 16.00,
    Quant.F16: 16.00,
}


class CacheQuant(str, Enum):
    """KV-cache element type (llama.cpp `-ctk` / `-ctv`, Ollama JSON `type_k`/`type_v`)."""

    F16 = "f16"
    Q8_0 = "q8_0"
    Q4_0 = "q4_0"

    @property
    def bytes_per_element(self) -> float:
        # Exact block math: q8_0 = 34 B / 32 el, q4_0 = 18 B / 32 el.
        return _CACHE_BYTES_PER_ELEMENT[self]


_CACHE_BYTES_PER_ELEMENT: Dict[CacheQuant, float] = {
    CacheQuant.F16: 2.0,
    CacheQuant.Q8_0: 1.0625,
    CacheQuant.Q4_0: 0.5625,
}


class Backend(str, Enum):
    """Inference backend; drives the empirical runtime-overhead estimate."""

    CUDA = "cuda"
    ROCM = "rocm"
    VULKAN = "vulkan"
    METAL = "metal"
    CPU = "cpu"

    @property
    def fixed_overhead_mb(self) -> int:
        """Fixed allocator/runtime overhead (handles, cublas, device buffers).

        Empirical: delta between model+KV size and total VRAM allocated
        right after model load in llama.cpp.
        """
        return _BACKEND_FIXED_OVERHEAD_MB[self]


_BACKEND_FIXED_OVERHEAD_MB: Dict[Backend, int] = {
    Backend.CUDA: 450,
    Backend.ROCM: 650,
    Backend.VULKAN: 750,
    Backend.METAL: 300,
    Backend.CPU: 0,
}


# ---------------------------------------------------------------------------
# Model architecture presets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPreset:
    """Geometry of a model family, enough for exact KV-cache math.

    Field values come from the HuggingFace `config.json` of the reference
    checkpoint (e.g. meta-llama/Meta-Llama-3-8B, Qwen/Qwen2.5-32B-Instruct).
    """

    key: str
    name: str
    params_b: float  # parameter count, billions (incl. embeddings)
    n_layers: int
    n_heads: int
    n_kv_heads: int  # KV heads after GQA grouping (== n_heads for MHA)
    head_dim: int
    hidden_size: int
    vocab_size: int
    ctx_train: int  # native max context (without RoPE scaling / YaRN)
    tied_embeddings: bool = False  # tok_embd doubles as output head
    mla_kv_dim: Optional[int] = None  # MLA compressed KV dim (kv_lora_rank + rope), DeepSeek
    aliases: Tuple[str, ...] = ()
    notes: str = ""

    @property
    def attention(self) -> str:
        if self.mla_kv_dim is not None:
            return "MLA"
        if self.n_kv_heads < self.n_heads:
            return "GQA"
        return "MHA"

    @property
    def is_mla(self) -> bool:
        return self.mla_kv_dim is not None

    def kv_bytes_per_token(self, cache_quant: CacheQuant = CacheQuant.F16) -> float:
        """KV-cache bytes consumed by ONE token in ONE layer.

        Standard MHA/GQA caches a K and a V vector per KV head:
            2 * n_kv_heads * head_dim * bytes_per_element
        MLA (DeepSeek-V3/R1) caches the single compressed latent
        c_KV (512) + decoupled RoPE keys (64) = 576 elements, shared by
        all 128 attention heads -- hence no factor of 2 and no head count.
        """
        bpe = cache_quant.bytes_per_element
        if self.mla_kv_dim is not None:
            return self.mla_kv_dim * bpe
        return 2 * self.n_kv_heads * self.head_dim * bpe


_MODELS: List[ModelPreset] = [
    # --- Llama family ------------------------------------------------------
    ModelPreset(
        key="llama-3-8b",
        name="Llama 3 8B",
        params_b=8.03,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        hidden_size=4096,
        vocab_size=128256,
        ctx_train=8192,
        aliases=("llama3-8b", "llama-3.0-8b", "meta-llama-3-8b"),
    ),
    ModelPreset(
        key="llama-3-70b",
        name="Llama 3 70B",
        params_b=70.55,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        head_dim=128,
        hidden_size=8192,
        vocab_size=128256,
        ctx_train=8192,
        aliases=("llama3-70b", "llama-3.0-70b", "meta-llama-3-70b"),
    ),
    ModelPreset(
        key="llama-3.1-8b",
        name="Llama 3.1 8B",
        params_b=8.03,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        hidden_size=4096,
        vocab_size=128256,
        ctx_train=131072,
        aliases=("llama3.1-8b", "llama-3.1-8b-instruct"),
    ),
    ModelPreset(
        key="llama-3.1-70b",
        name="Llama 3.1 70B",
        params_b=70.55,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        head_dim=128,
        hidden_size=8192,
        vocab_size=128256,
        ctx_train=131072,
        aliases=("llama3.1-70b", "llama-3.1-70b-instruct"),
    ),
    ModelPreset(
        key="llama-2-7b",
        name="Llama 2 7B",
        params_b=6.74,
        n_layers=32,
        n_heads=32,
        n_kv_heads=32,  # no GQA: full multi-head attention
        head_dim=128,
        hidden_size=4096,
        vocab_size=32000,
        ctx_train=4096,
        aliases=("llama2-7b", "llama-2-7b-chat"),
        notes="MHA -- 4x the KV cache of GQA models of the same size.",
    ),
    # --- Qwen 2.5 -----------------------------------------------------------
    ModelPreset(
        key="qwen2.5-7b",
        name="Qwen2.5 7B",
        params_b=7.62,
        n_layers=28,
        n_heads=28,
        n_kv_heads=4,
        head_dim=128,
        hidden_size=3584,
        vocab_size=152064,
        ctx_train=32768,
        tied_embeddings=True,
        aliases=("qwen-2.5-7b", "qwen2.5-7b-instruct"),
    ),
    ModelPreset(
        key="qwen2.5-14b",
        name="Qwen2.5 14B",
        params_b=14.77,
        n_layers=48,
        n_heads=40,
        n_kv_heads=8,
        head_dim=128,
        hidden_size=5120,
        vocab_size=152064,
        ctx_train=32768,
        aliases=("qwen-2.5-14b", "qwen2.5-14b-instruct"),
    ),
    ModelPreset(
        key="qwen2.5-32b",
        name="Qwen2.5 32B",
        params_b=32.76,
        n_layers=64,
        n_heads=40,
        n_kv_heads=8,
        head_dim=128,
        hidden_size=5120,
        vocab_size=152064,
        ctx_train=32768,
        aliases=("qwen-2.5-32b", "qwen2.5-32b-instruct"),
    ),
    ModelPreset(
        key="qwen2.5-72b",
        name="Qwen2.5 72B",
        params_b=72.71,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        head_dim=128,
        hidden_size=8192,
        vocab_size=152064,
        ctx_train=32768,
        aliases=("qwen-2.5-72b", "qwen2.5-72b-instruct"),
    ),
    # --- Qwen 3 (community favorites) ---------------------------------------
    ModelPreset(
        key="qwen3-8b",
        name="Qwen3 8B",
        params_b=8.19,
        n_layers=36,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        hidden_size=4096,
        vocab_size=151936,
        ctx_train=32768,
        tied_embeddings=True,
        aliases=("qwen-3-8b", "qwen3-8b-instruct"),
    ),
    ModelPreset(
        key="qwen3-32b",
        name="Qwen3 32B",
        params_b=32.76,
        n_layers=64,
        n_heads=40,
        n_kv_heads=8,
        head_dim=128,
        hidden_size=5120,
        vocab_size=151936,
        ctx_train=32768,
        aliases=("qwen-3-32b", "qwen3-32b-instruct"),
    ),
    # --- Mistral / Mixtral ---------------------------------------------------
    ModelPreset(
        key="mistral-7b",
        name="Mistral 7B v0.1",
        params_b=7.25,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        hidden_size=4096,
        vocab_size=32000,
        ctx_train=32768,
        aliases=("mistral-7b-v0.1", "mistral-7b-instruct"),
        notes="Uses sliding-window attention; real KV usage below 32k is smaller than shown.",
    ),
    ModelPreset(
        key="mixtral-8x7b",
        name="Mixtral 8x7B MoE",
        params_b=46.7,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        hidden_size=4096,
        vocab_size=32000,
        ctx_train=32768,
        aliases=("mixtral-8x7b-instruct", "mixtral"),
        notes="MoE: expert tensors are quantized harder in GGUF mixes; estimate is ~5-9% conservative.",
    ),
    # --- DeepSeek (MLA) -------------------------------------------------------
    ModelPreset(
        key="deepseek-v3",
        name="DeepSeek-V3 (671B MoE)",
        params_b=671.0,
        n_layers=61,
        n_heads=128,
        n_kv_heads=0,  # unused under MLA
        head_dim=0,  # unused under MLA
        hidden_size=7168,
        vocab_size=129280,
        ctx_train=131072,
        mla_kv_dim=512 + 64,  # kv_lora_rank + qk_rope_head_dim = 576 elements
        aliases=("deepseek-r1", "deepseek-v3-r1", "deepseek-r1-671b"),
        notes="MLA: KV cache is the 576-elem compressed latent. Quantized -ctk/-ctv is not supported by llama.cpp for MLA.",
    ),
    # --- Gemma 2 (very large vocab + head_dim, popular gotcha) ----------------
    ModelPreset(
        key="gemma-2-9b",
        name="Gemma 2 9B",
        params_b=9.24,
        n_layers=42,
        n_heads=16,
        n_kv_heads=8,
        head_dim=256,
        hidden_size=3584,
        vocab_size=256128,
        ctx_train=8192,
        tied_embeddings=True,
        aliases=("gemma2-9b", "gemma-2-9b-it"),
        notes="head_dim 256 and a 256k vocab make its KV + output head unusually fat.",
    ),
]

MODELS: Dict[str, ModelPreset] = {m.key: m for m in _MODELS}


def _normalize(text: str) -> str:
    return text.lower().replace("_", "").replace("-", "").replace(".", "").replace(" ", "")


def resolve_model(query: str) -> ModelPreset:
    """Resolve a user-supplied model name to a preset (fuzzy, case-insensitive)."""
    q = query.strip()
    for m in _MODELS:
        if q == m.key or q in m.aliases:
            return m
    pool = {m.key: _normalize(m.key) for m in _MODELS}
    for m in _MODELS:
        for alias in m.aliases:
            pool.setdefault(f"{m.key}:{alias}", _normalize(alias))
    nq = _normalize(q)
    for ident, norm in pool.items():
        if nq == norm:
            return MODELS[ident.split(":")[0]]
    matches = difflib.get_close_matches(nq, list(pool.values()), n=1, cutoff=0.75)
    if matches:
        ident = next(k for k, v in pool.items() if v == matches[0])
        return MODELS[ident.split(":")[0]]
    raise KeyError(
        f"Unknown model {query!r}. Known presets: {', '.join(sorted(MODELS))} "
        f"(or pass --params/--layers/... for a custom architecture)."
    )


def resolve_quant(query: str) -> Quant:
    q = query.strip().upper().replace("-", "_")
    try:
        return Quant(q)
    except ValueError:
        pass
    matches = difflib.get_close_matches(q, [v.value for v in Quant], n=1, cutoff=0.6)
    if matches:
        return Quant(matches[0])
    raise KeyError(f"Unknown quant {query!r}. Known: {', '.join(v.value for v in Quant)}")


def resolve_cache_quant(query: str) -> CacheQuant:
    q = query.strip().lower().replace("-", "_")
    try:
        return CacheQuant(q)
    except ValueError:
        raise KeyError(f"Unknown cache quant {query!r}. Known: f16, q8_0, q4_0")


def resolve_backend(query: str) -> Backend:
    q = query.strip().lower()
    try:
        return Backend(q)
    except ValueError:
        raise KeyError(f"Unknown backend {query!r}. Known: cuda, rocm, vulkan, metal, cpu")


def parse_context(value: Union[int, str]) -> int:
    """Accept 8192 or '8k' / '32K' / '128k' style context sizes."""
    if isinstance(value, int):
        ctx = value
    else:
        s = str(value).strip().lower().replace(" ", "").replace("_", "")
        mult = 1
        if s.endswith("k"):
            mult, s = 1024, s[:-1]
        elif s.endswith("m"):
            mult, s = 1024 * 1024, s[:-1]
        try:
            ctx = int(float(s) * mult)
        except ValueError:
            raise ValueError(f"Invalid context {value!r}; use e.g. 8192 or 32k")
    if ctx <= 0:
        raise ValueError("Context must be positive")
    return ctx


def custom_model(
    params_b: float,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    hidden_size: int,
    vocab_size: int,
    ctx_train: int = 32768,
    tied_embeddings: bool = False,
    mla_kv_dim: Optional[int] = None,
    name: str = "custom",
) -> ModelPreset:
    """Build a preset from a HuggingFace config.json of an unsupported model."""
    return ModelPreset(
        key="custom",
        name=name,
        params_b=params_b,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        ctx_train=ctx_train,
        tied_embeddings=tied_embeddings,
        mla_kv_dim=mla_kv_dim,
    )


# ---------------------------------------------------------------------------
# The estimate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Estimate:
    """Full VRAM breakdown for one (model, quant, context, backend) combo."""

    preset: ModelPreset
    quant: Quant
    context_len: int
    cache_quant: CacheQuant
    backend: Backend
    flash_attn: bool

    weights_gb: float
    kv_cache_gb: float
    overhead_gb: float
    total_gb: float
    # details
    kv_bytes_per_token: float
    fixed_overhead_gb: float
    scratch_gb: float
    over_train: bool  # requested context exceeds native training length

    @property
    def attention(self) -> str:
        return self.preset.attention

    def to_dict(self) -> dict:
        return {
            "model": self.preset.key,
            "model_name": self.preset.name,
            "attention": self.attention,
            "params_b": self.preset.params_b,
            "quant": self.quant.value,
            "bpw": self.quant.bpw,
            "context": self.context_len,
            "cache_quant": self.cache_quant.value,
            "backend": self.backend.value,
            "flash_attn": self.flash_attn,
            "kv_bytes_per_token": round(self.kv_bytes_per_token, 1),
            "vram_gb": {
                "weights": round(self.weights_gb, 2),
                "kv_cache": round(self.kv_cache_gb, 2),
                "overhead": round(self.overhead_gb, 2),
                "total": round(self.total_gb, 2),
            },
            "over_train_context": self.over_train,
        }


def estimate_weights(preset: ModelPreset, quant: Quant) -> float:
    """Model tensor size in GiB.

    Implements the project formula:
        model_vram = (params_billion * bpw * 10^9 / 8) / 1024^3
    Effective bpw already includes the higher-precision embedding/output
    tensors, so no extra fudge factor is applied.
    """
    return to_gib(preset.params_b * 1e9 * quant.bpw / 8)


def weight_split(preset: ModelPreset, quant: Quant) -> Tuple[float, float, float]:
    """Split weights for layer-offload planning: (total_gb, head_gb, per_layer_gb).

    `head_gb` is the size of ONE of tok_embd / output.weight (the two big
    vocab-sized tensors). tok_embd goes to GPU with the first offloaded
    layer, output.weight only when *all* layers are offloaded (llama.cpp
    treats `output.*` as layer `n_layers`; -ngl accepts n_layers + 1).
    Approximation: embeddings/output at `quant.head_bpw`, everything else
    uniform. Accurate to a few % on real checkpoints.
    """
    total = estimate_weights(preset, quant)
    head = to_gib(preset.vocab_size * preset.hidden_size * quant.head_bpw / 8)
    n_head_tensors = 1 if preset.tied_embeddings else 2
    per_layer = max(
        (total - head * n_head_tensors) / preset.n_layers,
        total * 0.5 / preset.n_layers,
    )
    return total, head, per_layer


def estimate_kv_cache(preset: ModelPreset, context_len: int, cache_quant: CacheQuant) -> Tuple[float, float]:
    """KV-cache size in GiB and bytes-per-token across all layers."""
    per_token = preset.kv_bytes_per_token(cache_quant) * preset.n_layers
    return to_gib(per_token * context_len), per_token


def attention_workspace(preset: ModelPreset, context_len: int, ubatch: int = 512) -> float:
    """fp16 attention-score workspace WITHOUT flash attention, in GiB.

    Without -fa, ggml materializes scores for one ubatch (512 tokens) of
    queries against the full context across all heads:
        ubatch * context * n_heads * 2 bytes
    For an 8B model at 8k this is 256 MiB; at 128k it explodes to 4+ GiB,
    which is exactly why 128k without -fa OOMs even 24 GB cards.
    With flash attention the workspace is not materialized -> 0.
    """
    return to_gib(ubatch * context_len * preset.n_heads * 2)


def context_overhead(
    preset: ModelPreset,
    context_len: int,
    backend: Backend,
    flash_attn: bool,
) -> Tuple[float, float]:
    """Empirical runtime overhead in GiB: (fixed_backend, scratch_buffers).

    Scratch = graph metadata + logits buffers (~96 MB) plus activations
    that scale with context x hidden size (~5 bytes/token/hidden-unit,
    calibrated against llama.cpp "compute buffer size" printouts), plus
    the attention workspace when flash attention is OFF (that one is what
    makes 128k contexts explode without -fa).
    """
    fixed = to_gib(backend.fixed_overhead_mb * MiB)
    base = 96 * MiB
    activation = context_len * preset.hidden_size * 5
    scores = 0 if flash_attn else attention_workspace(preset, context_len) * GiB
    scratch = to_gib(base + activation + scores)
    return fixed, scratch


def estimate_vram(
    model: Union[str, ModelPreset],
    quant: Union[str, Quant] = Quant.Q4_K_M,
    context_len: Union[int, str] = 8192,
    cache_quant: Union[str, CacheQuant] = CacheQuant.F16,
    backend: Union[str, Backend] = Backend.CUDA,
    flash_attn: bool = False,
) -> Estimate:
    """Estimate total VRAM needed for a fully-offloaded model. SDK entry point.

    >>> est = estimate_vram("llama-3-8b", "Q4_K_M", "8k", flash_attn=True)
    >>> round(est.total_gb, 1)
    6.2
    """
    preset = model if isinstance(model, ModelPreset) else resolve_model(model)
    q = quant if isinstance(quant, Quant) else resolve_quant(quant)
    cq = cache_quant if isinstance(cache_quant, CacheQuant) else resolve_cache_quant(cache_quant)
    be = backend if isinstance(backend, Backend) else resolve_backend(backend)
    ctx = parse_context(context_len)

    weights = estimate_weights(preset, q)
    kv, kv_per_token = estimate_kv_cache(preset, ctx, cq)
    fixed, scratch = context_overhead(preset, ctx, be, flash_attn)
    overhead = fixed + scratch
    total = weights + kv + overhead

    return Estimate(
        preset=preset,
        quant=q,
        context_len=ctx,
        cache_quant=cq,
        backend=be,
        flash_attn=flash_attn,
        weights_gb=weights,
        kv_cache_gb=kv,
        overhead_gb=overhead,
        total_gb=total,
        kv_bytes_per_token=kv_per_token,
        fixed_overhead_gb=fixed,
        scratch_gb=scratch,
        over_train=ctx > preset.ctx_train,
    )
