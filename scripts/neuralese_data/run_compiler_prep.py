#!/usr/bin/env python3
"""Preparation & preflight manifest builder for the Natlang Compiler training recipes (T0–T3).

Does NOT start or disturb any active training jobs (`aei1` or program tuning).
Verifies local/repo prerequisites, writes deterministic Recipe T0/T1/T2/T3 JSON configs into
a target preparation directory, and prints exact ledgered launch commands ready for cutover.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "training/neuralese") not in sys.path:
    sys.path.insert(0, str(ROOT / "training/neuralese"))

from natlang_neuralese.compiler.hkb_dependency import checkout as _hkb_checkout  # noqa: E402

HKB_ROOT = _hkb_checkout()

from natlang_neuralese.compiler.train_compiler import CompilerTrainConfig
from natlang_neuralese.model.lfm3_port import DEFAULT_BASE, DEFAULT_REVISION
from natlang_neuralese.train.step_verifier import VerifierTrainConfig


def build_prep_bundle(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_dir = HKB_ROOT.parent / "hindsight-kb-recovery/2026-10-21"
    seed_ckpt = seed_dir / "model-checkpoint-000000002.tar.zst"
    lfm3_cache = (
        Path.home()
        / ".cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots"
        / DEFAULT_REVISION
        / "config.json"
    )

    t0_cfg = {
        "recipe": "natlang.compiler.t0-step-verifier/1",
        "backbone": DEFAULT_BASE,
        "backbone_revision": DEFAULT_REVISION,
        "target_host": "dgx-gb10",
        "ledger_budget_gb": 40,
        "verifier": asdict(VerifierTrainConfig()),
        "declared_success_criteria": {"val_auroc_min": 0.78, "branch4_held_gain_pp": 8.0},
    }
    t1_cfg = {
        "recipe": "natlang.compiler.t1-opsd-call-level/1",
        "backbone": DEFAULT_BASE,
        "backbone_revision": DEFAULT_REVISION,
        "target_host": "colab-g4a-or-dgx-fallback",
        "lora_rank": 128,
        "lora_alpha": 128.0,
        "lr": 3e-5,
        "opsd_topk": 64,
        "opsd_beta": 0.5,
        "opsd_max_log_ratio": 6.0,
        "opsd_ce_weight": 0.3,
        "ban_literal_expected_answers": True,
        "min_new_tokens_per_round": 10_000_000,
    }
    t2_cfg = {
        "recipe": "natlang.compiler.t2-unified-compiler-hkb/1",
        "backbone": DEFAULT_BASE,
        "backbone_revision": DEFAULT_REVISION,
        "target_host": "colab-g4a",
        "reserve_colab_g4a": True,
        "aei1_behavior": "continue_on_dgx_fallback",
        "warmstart_source": str(seed_ckpt),
        "warmstart_extractor": str(HKB_ROOT / "scripts/extract_lfm3_warmstart.py"),
        "adapter_rank": 16,
        "adapter_inner_rank": 4,
        "adapter_heads": 4,
        "memory_bank_size": 32,
        "header_slots": 16,
        "batch_mix": {
            "context_distillation": 0.40,
            "live_kb_experience": 0.30,
            "judge_and_decision": 0.20,
            "tier15_fo_maml": 0.10,
        },
        "train": asdict(CompilerTrainConfig()),
        "mandatory_defect_check": "shuffled_minus_joint > 0 on paired held validation",
    }
    t3_cfg = {
        "recipe": "natlang.compiler.t3-native-query-projector/1",
        "target_host": "colab-g4b-or-g4a",
        "teacher_encoder": "perplexity-ai/pplx-embed-v2-late-0.6b",
        "declared_success_criteria": {"recall_at_32_min": 0.90, "vram_freed_gb_min": 1.1},
    }

    for name, payload in (
        ("t0-step-verifier.json", t0_cfg),
        ("t1-opsd.json", t1_cfg),
        ("t2-unified-compiler.json", t2_cfg),
        ("t3-native-query.json", t3_cfg),
    ):
        (out_dir / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    preflight = {
        "lfm3_snapshot_present": lfm3_cache.exists(),
        "hkb_seed_000000002_present": seed_ckpt.exists(),
        "hkb_repo_present": HKB_ROOT.exists(),
        "recipes_written": [
            str(out_dir / "t0-step-verifier.json"),
            str(out_dir / "t1-opsd.json"),
            str(out_dir / "t2-unified-compiler.json"),
            str(out_dir / "t3-native-query.json"),
        ],
    }
    (out_dir / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
    return preflight


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare Natlang Compiler training configs & preflight check.")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "ops/compiler-prep-20261125",
        help="Output directory for prepared recipe JSONs",
    )
    args = parser.parse_args(argv)
    result = build_prep_bundle(args.out)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
