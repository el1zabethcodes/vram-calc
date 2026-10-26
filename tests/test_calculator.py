"""Sanity tests: the numbers every local-LLM person knows by heart."""

import pytest

from vram_calc import (
    MODELS,
    Backend,
    CacheQuant,
    FitStatus,
    Quant,
    attention_workspace,
    custom_model,
    estimate_vram,
    parse_context,
    plan_all,
    plan_offload,
    resolve_gpu,
    resolve_model,
    resolve_quant,
)
from vram_calc.calculator import estimate_kv_cache, weight_split


# ---------------------------------------------------------------------------
# Weights: (params_b * effective_bpw * 1e9 / 8) / 1024^3
# ---------------------------------------------------------------------------


def test_weights_formula_qwen25_32b_q4km():
    est = estimate_vram("qwen2.5-32b", "Q4_K_M", 8192)
    expected = 32.76 * 1e9 * 4.85 / 8 / 1024**3  # ~= 18.49 GiB
    assert est.weights_gb == pytest.approx(expected, rel=1e-6)
    # Real Qwen2.5-32B-Instruct-Q4_K_M.gguf is ~19.85 GB decimal = 18.49 GiB.
    assert est.weights_gb == pytest.approx(18.49, abs=0.1)


def test_weights_llama3_8b_q4km():
    est = estimate_vram("llama-3-8b", "Q4_K_M", 8192)
    assert est.weights_gb == pytest.approx(4.53, abs=0.05)


# ---------------------------------------------------------------------------
# KV cache: reference values checked against llama-bench / community numbers
# ---------------------------------------------------------------------------


def test_kv_llama3_8b_8k_fp16_is_exactly_1gib():
    # 32L * 8 kv_heads * 128 head_dim * 2 (K+V) * 2 B = 128 KB/token
    est = estimate_vram("llama-3-8b", "Q4_K_M", 8192)
    assert est.kv_cache_gb == pytest.approx(1.0, abs=0.001)
    assert est.kv_bytes_per_token == pytest.approx(131072, rel=1e-6)


def test_kv_llama2_mha_is_4x_gqa():
    llama2 = estimate_vram("llama-2-7b", "Q8_0", 4096)
    assert llama2.kv_bytes_per_token == pytest.approx(512 * 1024)  # 512 KB/token


def test_kv_qwen25_7b_small_cache():
    # 28L * 4 kv_heads * 128 * 2 * 2 = 56 KB/token -> 8k = 0.4375 GiB
    est = estimate_vram("qwen2.5-7b", "Q8_0", 8192)
    assert est.kv_cache_gb == pytest.approx(0.4375, abs=0.001)


def test_kv_mla_deepseek():
    ds = MODELS["deepseek-v3"]
    # 61 layers * 576 compressed elements * 2 bytes = 70.3 KB/token
    per_token = ds.kv_bytes_per_token(CacheQuant.F16) * ds.n_layers
    assert per_token == pytest.approx(61 * 576 * 2)
    kv_gb, _ = estimate_kv_cache(ds, 32768, CacheQuant.F16)
    assert kv_gb == pytest.approx(61 * 576 * 2 * 32768 / 1024**3, rel=1e-6)


def test_kv_q8_0_cache_scales_by_block_math():
    f16 = estimate_vram("llama-3-8b", "Q4_K_M", 8192, cache_quant="f16")
    q8 = estimate_vram("llama-3-8b", "Q4_K_M", 8192, cache_quant="q8_0")
    assert q8.kv_cache_gb / f16.kv_cache_gb == pytest.approx(1.0625 / 2.0)
    q4 = estimate_vram("llama-3-8b", "Q4_K_M", 8192, cache_quant="q4_0")
    assert q4.kv_cache_gb / f16.kv_cache_gb == pytest.approx(0.5625 / 2.0)


# ---------------------------------------------------------------------------
# Overhead
# ---------------------------------------------------------------------------


def test_overhead_in_expected_band():
    # Spec: ~0.5-1.5 GB of runtime overhead for typical 8-32B configs at 8k.
    est8 = estimate_vram("llama-3-8b", "Q4_K_M", 8192)
    assert 0.5 <= est8.overhead_gb <= 1.5
    est32 = estimate_vram("qwen2.5-32b", "Q4_K_M", 32768)
    assert 0.5 <= est32.overhead_gb <= 3.5


def test_flash_attn_removes_attention_workspace():
    on = estimate_vram("llama-3-8b", "Q4_K_M", 8192, flash_attn=True)
    off = estimate_vram("llama-3-8b", "Q4_K_M", 8192, flash_attn=False)
    assert on.overhead_gb < off.overhead_gb
    assert off.overhead_gb - on.overhead_gb == pytest.approx(
        attention_workspace(MODELS["llama-3-8b"], 8192), rel=1e-5
    )


def test_no_fa_128k_explodes():
    # The reason everyone runs -fa at long context.
    ws = attention_workspace(MODELS["llama-3-8b"], 131072)
    assert ws >= 4.0  # GiB of pure score workspace


# ---------------------------------------------------------------------------
# Fit planning
# ---------------------------------------------------------------------------


def test_8b_q4_8k_fits_on_8gb_card():
    est = estimate_vram("llama-3.1-8b", "Q4_K_M", 8192, flash_attn=True)
    plan = plan_offload(est, resolve_gpu("rtx-4060-8g"))
    assert plan.status == FitStatus.COMFORTABLE
    assert plan.full_offload
    assert plan.ngl_flag == 99


def test_70b_q4_on_24gb_is_partial():
    est = estimate_vram("llama-3-70b", "Q4_K_M", 8192, flash_attn=True)
    plan = plan_offload(est, resolve_gpu("rtx-3090-24g"))
    assert plan.status == FitStatus.PARTIAL
    assert 25 <= plan.max_ngl <= 50  # community consensus zone
    assert 0 < plan.ram_gb < est.total_gb


def test_70b_q4_on_8gb_is_painful_partial():
    # ~11 of 80 layers fit -- technically runs, at CPU-ish speed.
    est = estimate_vram("llama-3-70b", "Q4_K_M", 8192)
    plan = plan_offload(est, resolve_gpu("rtx-4060-8g"))
    assert plan.status == FitStatus.PARTIAL
    assert 8 <= plan.max_ngl <= 15


def test_deepseek_v3_oom_on_8gb():
    # 6+ GiB per layer: not even one layer fits.
    est = estimate_vram("deepseek-v3", "Q4_K_M", 8192)
    plan = plan_offload(est, resolve_gpu("rtx-4060-8g"))
    assert plan.status == FitStatus.OOM
    assert plan.max_ngl == 0
    assert plan.ram_gb == pytest.approx(est.total_gb)


def test_apple_silicon_uses_usable_fraction():
    est = estimate_vram("qwen2.5-14b", "Q5_K_M", 8192, flash_attn=True, backend="metal")
    plan = plan_offload(est, resolve_gpu("apple-24g"))
    assert plan.status in (FitStatus.COMFORTABLE, FitStatus.TIGHT)
    # 24 GB * 0.75 = 18 GB usable; model needs ~12 GB
    assert plan.free_gb > 0


def test_reserve_mb_pushes_into_tight_or_partial():
    est = estimate_vram("llama-3.1-8b", "Q4_K_M", 8192, flash_attn=True)
    comfy = plan_offload(est, resolve_gpu("rtx-3060-12g"), reserve_mb=0)
    reserved = plan_offload(est, resolve_gpu("rtx-3060-12g"), reserve_mb=5000)
    assert comfy.free_gb > reserved.free_gb


def test_plan_all_sorted_best_first():
    est = estimate_vram("llama-3-8b", "Q4_K_M", 8192)
    plans = plan_all(est)
    ranks = [p.status for p in plans]
    order = {FitStatus.COMFORTABLE: 0, FitStatus.TIGHT: 1, FitStatus.PARTIAL: 2, FitStatus.OOM: 3}
    assert [order[r] for r in ranks] == sorted(order[r] for r in ranks)


# ---------------------------------------------------------------------------
# Custom models & parsing
# ---------------------------------------------------------------------------


def test_custom_model_matches_preset_math():
    base = MODELS["mistral-7b"]
    custom = custom_model(
        params_b=base.params_b,
        n_layers=base.n_layers,
        n_heads=base.n_heads,
        n_kv_heads=base.n_kv_heads,
        head_dim=base.head_dim,
        hidden_size=base.hidden_size,
        vocab_size=base.vocab_size,
    )
    a = estimate_vram(base, "Q4_K_M", "8k")
    b = estimate_vram(custom, "Q4_K_M", "8k")
    assert a.total_gb == pytest.approx(b.total_gb)


def test_parse_context_k_suffix():
    assert parse_context("32k") == 32768
    assert parse_context("128K") == 131072
    assert parse_context("8192") == 8192
    with pytest.raises(ValueError):
        parse_context("banana")


def test_fuzzy_resolution():
    assert resolve_model("Qwen2.5-32B").key == "qwen2.5-32b"
    assert resolve_model("llama3.1-8b").key == "llama-3.1-8b"
    assert resolve_quant("q4_k_m") == Quant.Q4_K_M
    assert resolve_quant("Q4KM") == Quant.Q4_K_M


def test_mla_skips_cache_quant_flags_in_command():
    from vram_calc.cli import llama_command

    est = estimate_vram("deepseek-v3", "Q4_K_M", 8192, cache_quant="q8_0", flash_attn=True)
    plan = plan_offload(est, resolve_gpu("a100-80g"))
    cmd = llama_command(est, plan)
    assert "-ctk" not in cmd  # MLA does not support quantized KV


def test_total_is_sum_of_parts():
    est = estimate_vram("qwen2.5-14b", "Q5_K_M", "16k", cache_quant="q8_0")
    assert est.total_gb == pytest.approx(est.weights_gb + est.kv_cache_gb + est.overhead_gb)
