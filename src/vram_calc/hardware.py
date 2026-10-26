"""GPU database and layer-offload fit planning.

The planner mirrors how llama.cpp actually distributes tensors with
`-ngl N`: tok_embd moves with the first offloaded layer, `output.*` only
when every layer is offloaded, and each layer carries its slice of the
KV cache. Compute buffers and backend overhead always stay on the GPU.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from .calculator import Estimate, weight_split


class FitStatus(str, Enum):
    COMFORTABLE = "comfortable"  # full offload with healthy headroom
    TIGHT = "tight"  # full offload, but you are one long prompt from OOM
    PARTIAL = "partial"  # some layers on GPU, rest on CPU
    OOM = "oom"  # not even one layer + KV + overhead fits


@dataclass(frozen=True)
class GPU:
    key: str
    name: str
    vram_mb: int  # total, as reported by nvidia-smi / About This Mac
    unified: bool = False  # Apple Silicon: shared with the OS
    usable_fraction: float = 1.0
    note: str = ""

    @property
    def vram_gb(self) -> float:
        return self.vram_mb / 1024


GPUS: List[GPU] = [
    # --- Desktop discrete ------------------------------------------------
    GPU(key="rtx-4060-8g", name="RTX 4060 8GB", vram_mb=8192),
    GPU(key="rtx-3060-12g", name="RTX 3060 12GB", vram_mb=12288),
    GPU(key="rtx-4060-ti-16g", name="RTX 4060 Ti 16GB", vram_mb=16384),
    GPU(key="rtx-4070-12g", name="RTX 4070 12GB", vram_mb=12288),
    GPU(key="rtx-5070-12g", name="RTX 5070 12GB", vram_mb=12288),
    GPU(key="rtx-3090-24g", name="RTX 3090 24GB", vram_mb=24576),
    GPU(key="rtx-4090-24g", name="RTX 4090 24GB", vram_mb=24576),
    GPU(key="rtx-5090-32g", name="RTX 5090 32GB", vram_mb=32768),
    GPU(key="p40-24g", name="Tesla P40 24GB", vram_mb=24576, note="No flash-attn, slow FP16; classic budget pick"),
    GPU(key="a100-80g", name="A100 80GB", vram_mb=81920),
    # --- Apple Silicon (unified memory) ------------------------------------
    # macOS caps a single process at ~75% of unified memory by default;
    # raise with `sudo sysctl iogpu.wired_limit_mb=<MiB>`.
    GPU(key="apple-16g", name="Apple Silicon 16GB", vram_mb=16384, unified=True, usable_fraction=0.75),
    GPU(key="apple-24g", name="Apple Silicon 24GB", vram_mb=24576, unified=True, usable_fraction=0.75),
    GPU(key="apple-36g", name="Apple Silicon 36GB", vram_mb=36864, unified=True, usable_fraction=0.75),
    GPU(key="apple-48g", name="Apple Silicon 48GB", vram_mb=49152, unified=True, usable_fraction=0.75),
    GPU(key="apple-64g", name="Apple Silicon 64GB", vram_mb=65536, unified=True, usable_fraction=0.75),
    GPU(key="apple-128g", name="Apple Silicon 128GB", vram_mb=131072, unified=True, usable_fraction=0.75),
]


def resolve_gpu(query: str) -> GPU:
    q = query.strip()
    for gpu in GPUS:
        if q == gpu.key or q.lower() == gpu.name.lower():
            return gpu
    pool = {g.key: g.key.replace("-", " ").lower() for g in GPUS}
    nq = q.lower().replace("-", " ").replace("_", " ")
    for key, norm in pool.items():
        if nq == norm:
            return next(g for g in GPUS if g.key == key)
    matches = difflib.get_close_matches(nq, list(pool.values()), n=1, cutoff=0.6)
    if matches:
        key = next(k for k, v in pool.items() if v == matches[0])
        return next(g for g in GPUS if g.key == key)
    raise KeyError(f"Unknown GPU {query!r}. Use --list-gpus to see the database.")


def custom_gpu(vram_mb: int, name: str = "Custom GPU") -> GPU:
    if vram_mb <= 0:
        raise ValueError("--vram-mb must be positive")
    return GPU(key="custom", name=name, vram_mb=vram_mb)


def usable_mb(gpu: GPU, reserve_mb: int = 0) -> float:
    """VRAM actually available to llama.cpp after OS/display reservation."""
    return max(gpu.vram_mb * gpu.usable_fraction - reserve_mb, 0.0)


# Kept back from every fit decision -- allocator rounding, fragmentation
# and the difference between our estimate and reality.
SAFETY_MARGIN_GB = 0.125


@dataclass
class OffloadPlan:
    """Result of fitting one estimate onto one GPU."""

    gpu: GPU
    status: FitStatus
    layers_total: int
    max_ngl: int  # recommended -ngl (== layers_total + 1 means full offload)
    full_total_gb: float  # VRAM needed for a 100% offload
    used_gb: float  # VRAM used at max_ngl
    free_gb: float  # usable - full_total (negative = deficit at 100% offload)
    ram_gb: float  # host RAM needed for the CPU-resident remainder
    reserve_mb: int
    free_at_max_gb: float = 0.0  # VRAM left at the recommended max_ngl
    notes: List[str] = field(default_factory=list)

    @property
    def full_offload(self) -> bool:
        return self.status in (FitStatus.COMFORTABLE, FitStatus.TIGHT)

    @property
    def ngl_flag(self) -> int:
        """What to actually pass to -ngl / num_gpu (99 == everything)."""
        return 99 if self.full_offload else self.max_ngl

    def to_dict(self) -> dict:
        return {
            "gpu": self.gpu.key,
            "gpu_name": self.gpu.name,
            "vram_mb": self.gpu.vram_mb,
            "status": self.status.value,
            "layers_total": self.layers_total,
            "max_ngl": self.max_ngl,
            "ngl_flag": self.ngl_flag,
            "full_total_gb": round(self.full_total_gb, 2),
            "used_gb": round(self.used_gb, 2),
            "free_gb": round(self.free_at_max_gb, 2),
            "deficit_at_full_gb": round(-min(self.free_gb, 0), 2),
            "ram_gb": round(self.ram_gb, 2),
            "notes": self.notes,
        }


def _tight_threshold_gb(usable_gb: float) -> float:
    """Headroom below which a full offload is called 'tight'.

    ~6% of the card, floor 768 MiB (Windows DWM alone eats half of that).
    """
    return max(0.75, 0.06 * usable_gb)


def plan_offload(est: Estimate, gpu: GPU, reserve_mb: int = 0) -> OffloadPlan:
    """Compute the best layer split for `est` on `gpu`.

    Mirrors llama.cpp tensor placement for `-ngl N`:
      * ngl >= 1          -> tok_embd on GPU
      * ngl >= layers + 1 -> output.weight (+ final norm) on GPU
      * KV moves per layer; compute buffer + backend overhead stay on GPU
    """
    preset = est.preset
    n_layers = preset.n_layers
    total_w, head, per_layer = weight_split(preset, est.quant)
    kv_per_layer = est.kv_cache_gb / n_layers
    usable = usable_mb(gpu, reserve_mb) / 1024.0  # -> GiB

    def gpu_need(ngl: int) -> float:
        if ngl <= 0:
            return 0.0
        need = head * min(ngl, 1)  # tok_embd
        need += per_layer * min(ngl, n_layers)
        if ngl >= n_layers + 1 and not preset.tied_embeddings:
            need += head  # output.weight joins the last GPU buffer
        need += kv_per_layer * min(ngl, n_layers)
        need += est.overhead_gb
        return need

    full_ngl = n_layers + (0 if preset.tied_embeddings else 1)
    full_total = gpu_need(full_ngl)
    notes: List[str] = [gpu.note] if gpu.note else []
    budget = usable - SAFETY_MARGIN_GB

    if full_total <= budget:
        free = usable - full_total
        status = FitStatus.TIGHT if free < _tight_threshold_gb(usable) else FitStatus.COMFORTABLE
        max_ngl = full_ngl
        used = full_total
        ram = 0.0
    else:
        best = 0
        for ngl in range(1, n_layers + 1):
            if gpu_need(ngl) <= budget:
                best = ngl
            else:
                break
        max_ngl = best
        free = usable - full_total  # negative: the deficit at 100% offload
        # A split that leaves <5% of layers (min 2) on the GPU is pointless:
        # it runs at CPU speed either way. Call it OOM and say why.
        min_useful = max(2, -(-n_layers // 20))
        if max_ngl < min_useful:
            status = FitStatus.OOM
            used = 0.0
            ram = est.total_gb  # everything lives in system RAM
            if max_ngl:
                notes.append(
                    f"Only {max_ngl}/{n_layers} layers would fit (<5%) -- GPU offload is not "
                    "worth it; run on CPU or drop quant/context."
                )
            else:
                notes.append(
                    "Even 1 layer + KV + compute buffer does not fit; run on CPU or drop quant/context."
                )
        else:
            status = FitStatus.PARTIAL
            used = gpu_need(max_ngl)
            # CPU-resident remainder: weights not on GPU + KV of CPU layers.
            cpu_layers = n_layers - max_ngl
            cpu_head_out = head if not preset.tied_embeddings else 0.0
            ram = per_layer * cpu_layers + kv_per_layer * cpu_layers + cpu_head_out
            notes.append(
                f"Partial offload: {max_ngl}/{n_layers} layers on GPU, "
                f"{cpu_layers} on CPU -- expect noticeably slower prompt processing."
            )

    return OffloadPlan(
        gpu=gpu,
        status=status,
        layers_total=n_layers,
        max_ngl=max_ngl,
        full_total_gb=full_total,
        used_gb=used,
        free_gb=free,
        ram_gb=ram,
        reserve_mb=reserve_mb,
        free_at_max_gb=usable - used,
        notes=notes,
    )


def plan_all(est: Estimate, reserve_mb: int = 0, gpus: Optional[List[GPU]] = None) -> List[OffloadPlan]:
    """Plan the estimate against every GPU in the database (best first)."""
    plans = [plan_offload(est, gpu, reserve_mb) for gpu in (gpus or GPUS)]
    order = {FitStatus.COMFORTABLE: 0, FitStatus.TIGHT: 1, FitStatus.PARTIAL: 2, FitStatus.OOM: 3}
    # Secondary: most free VRAM first (for partials, that means most layers).
    return sorted(plans, key=lambda p: (order[p.status], -p.free_gb, -p.max_ngl))
