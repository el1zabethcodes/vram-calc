"""Interactive CLI for vram-calc.

Usage patterns::

    vram-calc --model qwen2.5-32b --ctx 32k --quant Q4_K_M --gpu rtx-4090-24g
    vram-calc                      # interactive wizard
    vram-calc --list-models        # reference tables
    vram-calc ... --json           # machine-readable output
"""

from __future__ import annotations

import argparse
import json as jsonlib
import sys
from typing import List, Optional, Sequence

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from . import __version__
from .calculator import (
    MODELS,
    Quant,
    attention_workspace,
    custom_model,
    estimate_vram,
    parse_context,
    resolve_backend,
    resolve_cache_quant,
    resolve_model,
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
)

STATUS_STYLE = {
    FitStatus.COMFORTABLE: "bold green",
    FitStatus.TIGHT: "bold yellow",
    FitStatus.PARTIAL: "bold cyan",
    FitStatus.OOM: "bold red",
}

STATUS_LABEL = {
    FitStatus.COMFORTABLE: "✓ FITS",
    FitStatus.TIGHT: "▲ TIGHT",
    FitStatus.PARTIAL: "◇ SPLIT",
    FitStatus.OOM: "✗ OOM",
}

_GB = "{:.2f} GB".format


def _fmt_gb(value: float) -> str:
    return f"{value:.2f} GB"


def _fmt_ctx(ctx: int) -> str:
    return f"{ctx / 1024:g}k" if ctx >= 1024 and ctx % 1024 == 0 else str(ctx)


def _model_slug(est) -> str:
    return f"{est.preset.key}-{est.quant.value.lower()}.gguf"


# ---------------------------------------------------------------------------
# Ready-to-run configs
# ---------------------------------------------------------------------------


def llama_command(est, plan: Optional[OffloadPlan]) -> str:
    """Ready-to-paste llama-server command line.

    `plan=None` means CPU-only (no fitting GPU): -ngl 0.
    """
    ngl = plan.ngl_flag if plan else 0
    parts = [
        "llama-server",
        "-m", f"./{_model_slug(est)}",
        "-c", str(est.context_len),
        "-ngl", str(ngl),
    ]
    if est.cache_quant.value != "f16" and not est.preset.is_mla:
        parts += ["-ctk", est.cache_quant.value, "-ctv", est.cache_quant.value]
    if est.flash_attn:
        parts.append("-fa")
    return " ".join(parts)


def modelfile(est, plan: Optional[OffloadPlan]) -> str:
    """Ollama Modelfile for the same configuration."""
    ngl = plan.ngl_flag if plan else 0
    lines = [f"FROM ./{_model_slug(est)}", "", f"PARAMETER num_ctx {est.context_len}"]
    if ngl > 0:
        lines.append(f"PARAMETER num_gpu {ngl}")
    if plan and not plan.full_offload:
        lines.append("")
        lines.append("# If you still hit OOM, drop num_gpu further:")
        lines.append(f"# PARAMETER num_gpu {max(plan.max_ngl - 8, 0)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _tips(est, plan: Optional[OffloadPlan]) -> List[str]:
    tips: List[str] = []
    if est.preset.notes:
        tips.append(est.preset.notes)
    if est.over_train:
        tips.append(
            f"Requested context {_fmt_ctx(est.context_len)} exceeds the native "
            f"{_fmt_ctx(est.preset.ctx_train)}; enable YaRN (--rope-scaling yarn) or expect quality loss."
        )
    if est.preset.is_mla and est.cache_quant.value != "f16":
        tips.append(
            "MLA models do not support -ctk/-ctv quantized KV in llama.cpp; the command omits them."
        )
    if not est.flash_attn:
        scores = attention_workspace(est.preset, est.context_len)
        tips.append(
            f"Flash attention (-fa) is off: the attention workspace alone adds ~{_fmt_gb(scores)}. "
            "Add -fa to reclaim it."
        )
    if plan and plan.gpu.unified:
        tips.append(
            f"{plan.gpu.name}: macOS caps a process at ~75% of unified memory. "
            "Raise it with: sudo sysctl iogpu.wired_limit_mb=<MiB>"
        )
    return [t for t in tips if t]


def _render_breakdown(console: Console, est) -> None:
    kv_note = (
        f"{est.kv_bytes_per_token / 1024:g} KB/token x {_fmt_ctx(est.context_len)} tokens"
        f" · {est.attention} · {est.preset.n_layers}L · {est.cache_quant.value}"
    )
    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold dim",
        pad_edge=False,
    )
    table.add_column("Component", style="bold", min_width=12)
    table.add_column("Size", justify="right", min_width=10)
    table.add_column("Details", style="dim")
    table.add_row("Weights", _fmt_gb(est.weights_gb), f"{est.preset.params_b:g}B × {est.quant.value} ({est.quant.bpw:.2f} bpw)")
    table.add_row("KV cache", _fmt_gb(est.kv_cache_gb), kv_note)
    table.add_row(
        "Overhead",
        _fmt_gb(est.overhead_gb),
        f"{est.backend.value} runtime + compute buffers"
        + (" · flash-attn" if est.flash_attn else ""),
    )
    table.add_row(Text("Total (100% offload)", style="bold"), Text(_fmt_gb(est.total_gb), style="bold cyan"), style="bold")
    console.print(table)


def _render_fit_table(console: Console, est, plans: Sequence[OffloadPlan]) -> None:
    table = Table(
        title="GPU fit",
        title_justify="left",
        title_style="bold",
        box=box.SIMPLE_HEAVY,
        pad_edge=False,
    )
    table.add_column("GPU", style="bold", min_width=16)
    table.add_column("VRAM", justify="right")
    table.add_column("Free", justify="right")
    table.add_column("Status", justify="left", min_width=14)
    table.add_column("Recommended -ngl", justify="right")
    for plan in plans:
        style = STATUS_STYLE[plan.status]
        if plan.status == FitStatus.OOM:
            rec = "—"
            free = f"{plan.free_gb:.1f}"  # negative: the deficit at 100% offload
        else:
            free = f"+{plan.free_at_max_gb:.1f}"
            rec = str(plan.ngl_flag) if plan.full_offload else (
                f"{plan.max_ngl}/{plan.layers_total} (RAM {_fmt_gb(plan.ram_gb)})"
            )
        table.add_row(
            plan.gpu.name,
            f"{plan.gpu.vram_gb:g} GB",
            free,
            Text(STATUS_LABEL[plan.status], style=style),
            rec,
        )
    console.print(table)


def _render_outputs(console: Console, est, plan: Optional[OffloadPlan]) -> None:
    cmd = llama_command(est, plan)
    console.print(
        Panel(
            Syntax(cmd, "bash", theme="ansi_dark", word_wrap=True),
            title="llama.cpp",
            title_align="left",
            box=box.ROUNDED,
            border_style="green",
        )
    )
    console.print(
        Panel(
            Syntax(modelfile(est, plan), "dockerfile", theme="ansi_dark", word_wrap=True),
            title="Ollama Modelfile  [dim](ollama create mymodel -f Modelfile)[/dim]",
            title_align="left",
            box=box.ROUNDED,
            border_style="blue",
        )
    )


def render_report(console: Console, est, plans: Sequence[OffloadPlan], reserve_mb: int) -> None:
    header = Table.grid(padding=(0, 2))
    header.add_row(
        Text(est.preset.name, style="bold cyan"),
        Text(f"{est.quant.value} · ctx {_fmt_ctx(est.context_len)} · {est.attention} · {est.backend.value}", style="dim"),
    )
    console.print(Panel(header, box=box.ROUNDED, border_style="cyan", padding=(0, 1)))
    _render_breakdown(console, est)
    console.print()
    if plans:
        _render_fit_table(console, est, plans)
    else:
        console.print(
            Panel(
                Text(
                    f"CPU backend: the whole model runs in system RAM. "
                    f"Budget ~{_fmt_gb(est.total_gb)} free memory.",
                    style="yellow",
                ),
                box=box.ROUNDED,
                border_style="dim",
            )
        )
    best = plans[0] if plans else None
    if best is not None and best.status == FitStatus.OOM:
        best = None
    console.print()
    _render_outputs(console, est, best)
    tips = _tips(est, plans[0] if plans else None)
    if tips:
        body = Text()
        for i, tip in enumerate(tips):
            body.append("• ", style="yellow")
            body.append(tip + "\n" if i < len(tips) - 1 else tip)
        body.rstrip()
        console.print(Panel(body, title="Notes", title_align="left", box=box.ROUNDED, border_style="dim"))


# ---------------------------------------------------------------------------
# List / reference output
# ---------------------------------------------------------------------------


def _print_models(console: Console) -> None:
    table = Table(title="Known model presets", box=box.SIMPLE_HEAVY, title_justify="left")
    table.add_column("Key", style="bold")
    table.add_column("Name")
    table.add_column("Params", justify="right")
    table.add_column("Layers", justify="right")
    table.add_column("Attn", justify="center")
    table.add_column("KV / token (fp16)", justify="right")
    table.add_column("Native ctx", justify="right")
    for m in sorted(MODELS.values(), key=lambda x: x.params_b):
        kv_kb = m.kv_bytes_per_token() * m.n_layers / 1024
        table.add_row(m.key, m.name, f"{m.params_b:g}B", str(m.n_layers), m.attention, f"{kv_kb:.0f} KB", _fmt_ctx(m.ctx_train))
    console.print(table)


def _print_quants(console: Console) -> None:
    table = Table(title="Quantization formats (effective bpw)", box=box.SIMPLE_HEAVY, title_justify="left")
    table.add_column("Quant", style="bold")
    table.add_column("Effective bpw", justify="right")
    table.add_column("8B model", justify="right")
    table.add_column("70B model", justify="right")
    for q in Quant:
        w8 = 8.03 * 1e9 * q.bpw / 8 / 1024**3
        w70 = 70.55 * 1e9 * q.bpw / 8 / 1024**3
        table.add_row(q.value, f"{q.bpw:.2f}", _fmt_gb(w8), _fmt_gb(w70))
    console.print(table)


def _print_gpus(console: Console) -> None:
    table = Table(title="GPU database", box=box.SIMPLE_HEAVY, title_justify="left")
    table.add_column("Key", style="bold")
    table.add_column("Name")
    table.add_column("VRAM", justify="right")
    table.add_column("Type", justify="center")
    table.add_column("Note", style="dim")
    for g in GPUS:
        table.add_row(g.key, g.name, f"{g.vram_gb:g} GB", "unified" if g.unified else "discrete", g.note)
    console.print(table)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vram-calc",
        description="Precise VRAM calculator for quantized GGUF models "
        "(weights + KV cache + runtime overhead), with ready-to-run "
        "llama.cpp and Ollama configs.",
        epilog="Run without arguments for the interactive wizard.",
    )
    parser.add_argument("--version", action="version", version=f"vram-calc {__version__}")
    parser.add_argument("-m", "--model", help="model preset key or alias (see --list-models)")
    parser.add_argument("-c", "--ctx", default="8192", help="context length: 8192 or '32k' [default: 8192]")
    parser.add_argument("-q", "--quant", default="Q4_K_M", help="weight quant (see --list-quants) [default: Q4_K_M]")
    parser.add_argument("--cache-quant", default="f16", help="KV cache type: f16 | q8_0 | q4_0 [default: f16]")
    parser.add_argument("--backend", default="cuda", help="cuda | rocm | vulkan | metal | cpu [default: cuda]")
    parser.add_argument("--flash-attn", action="store_true", help="flash attention enabled (smaller compute buffers)")
    parser.add_argument("--gpu", default="auto", help="GPU key from --list-gpus, 'auto' (matching kind) or 'all'")
    parser.add_argument("--vram-mb", type=int, help="custom GPU size in MiB (overrides --gpu)")
    parser.add_argument("--reserve-mb", type=int, default=0, help="MiB reserved by OS/display/browser [default: 0]")
    parser.add_argument("--json", action="store_true", help="machine-readable JSON output")
    # custom architecture
    custom = parser.add_argument_group("custom model (when the preset list is not enough)")
    custom.add_argument("--params", type=float, help="parameter count, billions")
    custom.add_argument("--layers", type=int, help="number of transformer layers")
    custom.add_argument("--heads", type=int, help="attention heads")
    custom.add_argument("--kv-heads", type=int, help="KV heads (GQA); equals --heads for MHA")
    custom.add_argument("--head-dim", type=int, help="head dimension")
    custom.add_argument("--hidden", type=int, help="hidden size")
    custom.add_argument("--vocab", type=int, help="vocabulary size")
    custom.add_argument("--mla-dim", type=int, help="MLA compressed KV dim (kv_lora_rank + rope), e.g. 576 for DeepSeek")
    custom.add_argument("--tied-embeddings", action="store_true", help="tied input/output embeddings")
    # listing
    listing = parser.add_argument_group("reference tables")
    listing.add_argument("--list-models", action="store_true")
    listing.add_argument("--list-quants", action="store_true")
    listing.add_argument("--list-gpus", action="store_true")
    return parser


def _gpus_for(args, backend: str) -> List[GPU]:
    if args.vram_mb:
        return [custom_gpu(args.vram_mb)]
    if args.gpu and args.gpu.lower() == "all":
        return GPUS
    if args.gpu and args.gpu.lower() not in ("auto",):
        return [resolve_gpu(args.gpu)]
    # auto: pick the family that matches the backend
    if backend == "metal":
        return [g for g in GPUS if g.unified]
    if backend == "cpu":
        return []
    return [g for g in GPUS if not g.unified]


def _build_estimate(args) -> object:
    backend = resolve_backend(args.backend)
    if args.params:
        missing = [
            name
            for name, val in (
                ("--layers", args.layers),
                ("--heads", args.heads),
                ("--kv-heads", args.kv_heads),
                ("--head-dim", args.head_dim),
                ("--hidden", args.hidden),
                ("--vocab", args.vocab),
            )
            if not val
        ]
        if missing:
            raise ValueError("Custom model requires: " + ", ".join(missing))
        preset = custom_model(
            params_b=args.params,
            n_layers=args.layers,
            n_heads=args.heads,
            n_kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            hidden_size=args.hidden,
            vocab_size=args.vocab,
            tied_embeddings=args.tied_embeddings,
            mla_kv_dim=args.mla_dim,
        )
    else:
        if not args.model:
            raise ValueError("No model given")
        preset = resolve_model(args.model)
    return estimate_vram(
        preset,
        quant=args.quant,
        context_len=parse_context(args.ctx),
        cache_quant=args.cache_quant,
        backend=backend,
        flash_attn=args.flash_attn,
    )


def _render_json(est, plans: Sequence[OffloadPlan]) -> str:
    payload = est.to_dict()
    payload["tool"] = {"name": "vram-calc", "version": __version__}
    payload["gpus"] = [p.to_dict() for p in plans]
    payload["llama_cpp"] = llama_command(est, plans[0] if plans else None)
    payload["modelfile"] = modelfile(est, plans[0] if plans else None)
    return jsonlib.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# Interactive wizard
# ---------------------------------------------------------------------------


def _wizard(console: Console):
    console.print(Panel("vram-calc interactive setup", box=box.ROUNDED, border_style="cyan"))

    # 1. model
    console.print("[bold]1/5 Model[/bold] -- pick a preset number, type a name, or 'custom'.")
    presets = sorted(MODELS.values(), key=lambda x: x.params_b)
    for i, m in enumerate(presets, 1):
        console.print(f"  [cyan]{i:>2}[/cyan]  {m.name:<22} [dim]({m.params_b:g}B, {m.attention})[/dim]")
    raw = Prompt.ask("Model", default="qwen2.5-7b").strip().lower()
    if raw in ("custom", "c"):
        params = Prompt.ask("  Parameter count (billions)", default="7.6")
        layers = IntPrompt.ask("  Layers", default=32)
        heads = IntPrompt.ask("  Attention heads", default=32)
        kv_heads = IntPrompt.ask("  KV heads (GQA)", default=8)
        head_dim = IntPrompt.ask("  Head dim", default=128)
        hidden = IntPrompt.ask("  Hidden size", default=4096)
        vocab = IntPrompt.ask("  Vocab size", default=32000)
        mla = IntPrompt.ask("  MLA compressed KV dim (0 = not MLA)", default=0)
        tied = Confirm.ask("  Tied embeddings?", default=False)
        preset = custom_model(
            params_b=float(params),
            n_layers=layers,
            n_heads=heads,
            n_kv_heads=kv_heads,
            head_dim=head_dim,
            hidden_size=hidden,
            vocab_size=vocab,
            tied_embeddings=tied,
            mla_kv_dim=mla or None,
        )
    elif raw.isdigit() and 1 <= int(raw) <= len(presets):
        preset = presets[int(raw) - 1]
    else:
        preset = resolve_model(raw)

    # 2. context
    console.print("\n[bold]2/5 Context length[/bold]")
    ctx = Prompt.ask("Context (e.g. 4k, 8k, 32k, 128k)", default="8k")

    # 3. quant
    console.print("\n[bold]3/5 Weight quantization[/bold]")
    quant = Prompt.ask("Quant", default="Q4_K_M")

    # 4. cache + flash attn
    console.print("\n[bold]4/5 KV cache[/bold]")
    cache = Prompt.ask("Cache type", choices=["f16", "q8_0", "q4_0"], default="f16")
    flash = Confirm.ask("Flash attention available?", default=True)

    # 5. hardware
    console.print("\n[bold]5/5 Hardware[/bold]")
    backend = Prompt.ask("Backend", choices=["cuda", "rocm", "vulkan", "metal", "cpu"], default="cuda")
    custom_vram = Confirm.ask("Enter GPU size manually?", default=False)
    vram_mb = IntPrompt.ask("VRAM (MiB, e.g. 12288)", default=12288) if custom_vram else 0
    reserve = IntPrompt.ask("Reserve for OS/display (MiB)", default=0)

    return {
        "preset": preset,
        "ctx": ctx,
        "quant": quant,
        "cache": cache,
        "flash": flash,
        "backend": backend,
        "vram_mb": vram_mb,
        "reserve": reserve,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    console = Console()
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        _print_models(console)
        return 0
    if args.list_quants:
        _print_quants(console)
        return 0
    if args.list_gpus:
        _print_gpus(console)
        return 0

    try:
        if args.model is None and not args.params:
            if not sys.stdin.isatty():
                parser.print_help()
                return 0
            w = _wizard(console)
            est = estimate_vram(
                w["preset"],
                quant=w["quant"],
                context_len=parse_context(w["ctx"]),
                cache_quant=w["cache"],
                backend=resolve_backend(w["backend"]),
                flash_attn=w["flash"],
            )
            reserve = w["reserve"]
            if w["vram_mb"]:
                plans = [plan_offload(est, custom_gpu(w["vram_mb"]), reserve)]
            elif w["backend"] == "cpu":
                plans = []
            elif w["backend"] == "metal":
                plans = plan_all(est, reserve, gpus=[g for g in GPUS if g.unified])
            else:
                plans = plan_all(est, reserve, gpus=[g for g in GPUS if not g.unified])
        else:
            est = _build_estimate(args)
            reserve = args.reserve_mb
            gpu_list = _gpus_for(args, est.backend.value)
            plans = plan_all(est, reserve, gpus=gpu_list) if gpu_list else []
    except (KeyError, ValueError) as exc:
        console.print(f"[bold red]error:[/bold red] {exc}")
        return 2

    if args.json:
        console.print_json(_render_json(est, plans))
        return 0

    render_report(console, est, plans, reserve)
    return 0


if __name__ == "__main__":
    sys.exit(main())
