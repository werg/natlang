"""S3 pilot on real S1 port records: phases A→F with the harness at the phase boundaries.

    python -m natlang_neuralese.train.pilot --out RUN_DIR --records DIR [--families a,b] [--scale 1.0]

Small by design (a few thousand records, short schedule, capped GPU memory): the point is to
learn whether the channel carries content on real data before a full-machine run. Ordinary
text for phases A–C comes from the training records' own source texts.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import time
from pathlib import Path

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ..data.records import read_records  # noqa: E402
from ..data.render import Renderer, render_record, span_examples  # noqa: E402
from ..eval.harness import run_harness  # noqa: E402
from ..model.heads import PortHeads  # noqa: E402
from ..model.lfm2_port import ControlTokens, PortBackbone, backbone_identity, load_backbone, load_conv_kernel  # noqa: E402
from .phases import pilot_phases, with_phase_steps  # noqa: E402
from .smoke import phase_curves  # noqa: E402
from .trainer import Trainer  # noqa: E402

DEFAULT_RECORDS = "/mnt/external/natlang-development-data/data/neuralese/port-records/samples-20261003-closed"
EVAL_SPLITS = ("test", "validation")


def load_family(directory: Path, family: str, labels: set[str]):
    rows = read_records([directory / f"{family}.port-records.jsonl"], imitation_only=False)
    return [r for r in rows if r.outcome_label in labels]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--records", default=DEFAULT_RECORDS)
    parser.add_argument("--families", default="qa_extractive,qa_multihop,tool_digest")
    parser.add_argument("--labels", default="gold,checked,teacher")
    parser.add_argument("--train-per-family", type=int, default=2000)
    parser.add_argument("--eval-per-family", type=int, default=48)
    parser.add_argument("--max-producer-tokens", type=int, default=1536)
    parser.add_argument("--max-target-tokens", type=int, default=192)
    parser.add_argument("--natlang-share", type=float, default=0.25)
    parser.add_argument("--cutoff", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-gb", type=float, default=float(os.environ.get("NATLANG_CUDA_MEMORY_GB", 20.0)))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stop-after-phase", choices=list("ABCDEF"), default="F")
    parser.add_argument("--student-lora", default=None,
                        help="the crisp student's LoRA (a PEFT directory), merged into the base before port training (S3 §1)")
    parser.add_argument("--eval-families", default=None,
                        help="families of the held-out harness (default: --families; --stream may add evaluation-only families)")
    parser.add_argument("--checkpoint-every", type=int, default=200, help="steps between resumable checkpoints")
    parser.add_argument("--harness-phases", default="C,D,F", help="phases whose boundary runs the held-out harness")
    parser.add_argument("--phase-batch", default="", help="per-phase batch sizes, e.g. D=4 (rehearsals in less memory)")
    parser.add_argument("--phase-steps", default="",
                        help="per-phase step counts, e.g. A=3000,B=3000,C=4000,D=30000,E=4000,F=6000 (overrides --scale "
                             "for those phases; ramps inside a phase scale with it)")
    parser.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw",
                        help="adamw continues the A-F pilot lineage; muon is for new run directories only")
    parser.add_argument("--stop-exploration", type=float, default=0.0,
                        help="phase-E behaviour mixture weight (importance-weighted); 0 keeps the original schedule")
    parser.add_argument("--stop-temperature", type=float, default=1.0)
    parser.add_argument("--span-lengths", default="16",
                        help="comma-separated span lengths for phases A-C, cycled across windows (16 = the original pilot)")
    parser.add_argument("--tokens-per-vector", type=float, default=0.0,
                        help="phase D writes ceil(source tokens / this) vectors, teacher-forced, with stop BCE at the end "
                             "(0 = the stop head chooses, as in the original pilot)")
    parser.add_argument("--fail-on-shortcut", action="store_true",
                        help="refuse to train a phase whose supervised lengths a counter can predict (data/shortcuts.py)")
    parser.add_argument("--stop-source", choices=["shallow", "final"], default="shallow",
                        help="stop head reads the sketch states (original) or the completed full-depth states")
    parser.add_argument("--stream", action="store_true",
                        help="stream training records (data/stream.py) instead of loading --train-per-family of each")
    parser.add_argument("--expected-data-summary", help="Refuse resume if reconstructed data differs from this saved summary")
    parser.add_argument("--backbone", choices=["lfm2", "maple"], default="lfm2",
                        help="maple: the Maple student (maple/student.py); markers 151669/151670, ternary phase-F adapters")
    parser.add_argument("--maple-model", default="/home/werg/data/models/maple-preview-bf16")
    parser.add_argument("--maple-state", default=None,
                        help="nested-state.pt of the Maple student (shared QAT adapters, scales, family members)")
    parser.add_argument("--lora-layers", default="",
                        help="phase-F layers, released in this order (default: the phase schedule's; Maple: 23,22,21,20)")
    parser.add_argument("--unqualified-smoke", action="store_true",
                        help="Maple only: allow the legacy marker/RMS heads for a memory/throughput smoke whose port is "
                             "never used (no foundation certificate)")
    args = parser.parse_args(argv)
    if args.backbone == "maple" and not args.unqualified_smoke:
        # New lineages start from the shared foundation (AGENTS.md, plans/neuralese/TRAINING_RECIPE.md): token identity
        # and a qualified causal output-state -> next-token-embedding projection, consumed by the raw token-aligned
        # runtime. This trainer still builds the legacy marker/RMS feedback heads, so it cannot train a Maple port.
        parser.error("Maple port training requires a foundation certificate (recipes/foundation-maple-v1.json) and the "
                     "raw token-aligned runtime; the legacy heads here are not qualified (use --unqualified-smoke for "
                     "a memory smoke only)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rng = random.Random(args.seed)
    if args.device == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.memory_gb * 2**30 / total))

    if args.backbone == "maple":
        from ..maple.maple_port import MaplePortBackbone
        from ..maple.student import load_student, student_identity
        from ..model.hf_port import qwen_controls

        model, tokenizer = load_student(args.maple_model, args.maple_state, device=args.device)
        identity = student_identity(args.maple_model, args.maple_state)
        torch.manual_seed(args.seed)
        backbone = MaplePortBackbone(model, qwen_controls(tokenizer))
        if not args.lora_layers:
            args.lora_layers = ",".join(str(i) for i in range(backbone.num_layers - 1, backbone.num_layers - 5, -1))
    else:
        model, tokenizer = load_backbone(lora=args.student_lora, dtype=torch.bfloat16, device="cpu")
        identity = backbone_identity(args.student_lora)
        model.to(args.device)
        torch.manual_seed(args.seed)
        backbone = PortBackbone(model, ControlTokens.from_tokenizer(tokenizer), conv_kernel=load_conv_kernel())
    heads = PortHeads(backbone, cutoff=args.cutoff, max_length=args.max_length,
                      stop_source=args.stop_source).to(args.device)
    renderer = Renderer(tokenizer, backbone.controls)

    directory = Path(args.records)
    labels = set(args.labels.split(","))
    train, evaluation, counts, texts = [], [], {}, []

    def fits(rendered):
        return (len(rendered.producer) <= args.max_producer_tokens and len(rendered.target) <= args.max_target_tokens
                and len(rendered.consumer_before) + len(rendered.consumer_after) <= args.max_producer_tokens)

    if args.stream:
        # Full-corpus mode: training records stream from a byte-offset index; only evaluation rows are read,
        # the first --eval-per-family that fit the bounds in file order.
        from ..data.stream import PortRecordStream

        def stream(seed):
            return PortRecordStream({f: directory / f"{f}.port-records.jsonl" for f in args.families.split(",")}, renderer,
                                    index_dir=out / "index", labels=labels, seed=seed, natlang_share=args.natlang_share,
                                    max_producer_tokens=args.max_producer_tokens, max_target_tokens=args.max_target_tokens,
                                    held_log=out / "held-records.jsonl")
        for family in (args.eval_families or args.families).split(","):
            kept_eval = []
            for record in read_records([directory / f"{family}.port-records.jsonl"], imitation_only=False):
                if len(kept_eval) >= args.eval_per_family:
                    break
                if record.split in EVAL_SPLITS and record.outcome_label in labels:
                    rendered = render_record(renderer, record, form="chat")
                    if fits(rendered):
                        kept_eval.append(rendered)
            counts[family] = {"eval": len(kept_eval)}
            evaluation += kept_eval
        train = stream(args.seed)
        # Span texts come from a separately seeded stream so drawing them does not move the training stream.
        texts = stream(args.seed + 1).source_texts(4000)
        counts["stream"] = train.state_dict()["families"]
    else:
        for family in args.families.split(","):
            rows = load_family(directory, family, labels)
            rng.shuffle(rows)
            kept_train, kept_eval = [], []
            for record in rows:
                want_train = record.split == "train" and len(kept_train) < args.train_per_family
                want_eval = record.split in EVAL_SPLITS and len(kept_eval) < args.eval_per_family
                if not (want_train or want_eval):
                    continue
                form = "natlang" if want_train and rng.random() < args.natlang_share else "chat"
                rendered = render_record(renderer, record, form=form)
                if not fits(rendered):
                    continue
                (kept_train if want_train else kept_eval).append(rendered)
                if want_train:
                    texts.append(record.source_text())
            counts[family] = {"train": len(kept_train), "eval": len(kept_eval), "available": len(rows)}
            train += kept_train
            evaluation += kept_eval
        rng.shuffle(train)
        rng.shuffle(texts)
    span_lengths = [int(x) for x in args.span_lengths.split(",")]
    spans_train = list(span_examples(renderer, texts[: 4000], 64, 16, 32, limit=6000, span_lengths=span_lengths))
    eval_texts = [r.source_text() for r in read_records(
        [directory / f"{f}.port-records.jsonl" for f in args.families.split(",")], imitation_only=False)
        if r.split in EVAL_SPLITS][:200]
    spans_eval = list(span_examples(renderer, eval_texts, 64, 16, 32, limit=48, span_lengths=span_lengths))
    data_summary = {"families": counts, "spans_train": len(spans_train), "spans_eval": len(spans_eval)}
    if args.expected_data_summary and json.loads(Path(args.expected_data_summary).read_text()) != data_summary:
        raise ValueError("Reconstructed pilot data differs from the saved summary; refusing resume")
    (out / "data_summary.json").write_text(json.dumps(data_summary, indent=2))
    print(json.dumps(counts), len(spans_train), "spans", flush=True)

    reports = {}

    def harness(label):
        report = run_harness(backbone, heads, span_examples=spans_eval, rendered_records=evaluation,
                             cache_prefixes=[s.prefix for s in spans_eval[:4]], latency_prefix=evaluation[0].producer[:-1],
                             max_length=args.max_length, temperatures=(0.3, 1.0),
                             out_path=out / f"harness_{label}.json", label=label)
        reports[label] = report
        print(f"harness {label}: {time.time() - started:.0f}s", flush=True)
        return report

    phases = pilot_phases(args.scale, max_length=args.max_length)
    if args.phase_batch:
        sizes = {k: int(v) for k, v in (item.split("=") for item in args.phase_batch.split(","))}
        phases = [dataclasses.replace(p, batch_size=sizes[p.name]) if p.name in sizes else p for p in phases]
    if args.phase_steps:
        phases = with_phase_steps(phases, {k: int(v) for k, v in (item.split("=") for item in args.phase_steps.split(","))})
    phases = [dataclasses.replace(p, stop_exploration=args.stop_exploration, stop_temperature=args.stop_temperature)
              if p.name == "E" else p for p in phases]
    if args.lora_layers:
        layers = tuple(int(i) for i in args.lora_layers.split(","))
        phases = [dataclasses.replace(p, lora_layers=layers) if p.lora_layers else p for p in phases]
    if args.tokens_per_vector > 0:
        phases = [dataclasses.replace(p, tokens_per_vector=args.tokens_per_vector, stop_weight=1.0)
                  if p.name == "D" else p for p in phases]
    boundaries = {name: f"after_{name}" for name in args.harness_phases.split(",") if name}
    for end in range(len(phases)):
        # Restore against the full schedule: a phase-F checkpoint carries LoRA
        # optimizer groups even while revisiting earlier harness boundaries.
        trainer = Trainer(backbone, heads, phases, out, span_train=spans_train, records_train=train,
                          seed=args.seed, checkpoint_every=args.checkpoint_every, log=lambda m: print(m, flush=True),
                          stop_after_phase=phases[end].name, optimizer=args.optimizer,
                          fail_on_shortcut=args.fail_on_shortcut, backbone_identity=identity)
        trainer.run()
        if trainer._stop_requested:
            print("stopped on SIGTERM; rerun with the same --out to resume", flush=True)
            return
        name = phases[end].name
        at_boundary = trainer.phase_index == end + 1 and trainer.phase_step == 0
        print(f"phase {name} {'done' if at_boundary else 'already passed'}: "
              f"{time.time() - started:.0f}s, step {trainer.global_step}", flush=True)
        kept = out / f"checkpoint-after-{name}.pt"
        if at_boundary and not kept.exists() and (out / "checkpoint.pt").exists():
            # Each phase's end state stays available (the plan compares phases per family before accepting F).
            import shutil
            shutil.copyfile(out / "checkpoint.pt", kept)
        if name in boundaries:
            report_path = out / f"harness_{boundaries[name]}.json"
            if report_path.exists():
                reports[boundaries[name]] = json.loads(report_path.read_text())
            elif at_boundary:
                harness(boundaries[name])
            else:
                print(f"missing {boundaries[name]} report; later weights cannot reconstruct it", flush=True)
        if name == args.stop_after_phase:
            print(f"requested review boundary after phase {name}; later phases remain paused", flush=True)
            break

    summary = {"wall_seconds": time.time() - started, "data": counts,
               "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30 if args.device == "cuda" else None,
               "curves": phase_curves(out / "metrics.jsonl", window=25),
               "reports": {k: brief(v) for k, v in reports.items()}}
    (out / "pilot_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary["reports"], indent=2, default=str))


def brief(report: dict) -> dict:
    records = report.get("records", {})
    out = {"spans": {k: report.get("spans", {}).get(k) for k in ("nll", "correct_minus_shuffled", "gap_recovered")},
           "families": {f: {k: r.get(k) for k in ("nll", "correct_minus_shuffled", "correct_better_than_shuffled",
                                                    "gap_recovered")}
                        for f, r in records.get("families", {}).items()},
           "stopping": {k: records.get("stopping", {}).get(k) for k in ("mean_length", "truncation_rate",
                                                                         "length_one_rate", "length_source_spearman",
                                                                         "mean_source_tokens_per_vector")},
           "representation": records.get("representation"),
           "cache_agreement": report.get("cache_agreement"),
           "temperature_sweep": report.get("temperature_sweep")}
    return out


if __name__ == "__main__":
    main()
