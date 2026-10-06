"""End-to-end smoke run: phases A→B→C→D on small fixtures, harness before and after.

    python -m natlang_neuralese.train.smoke --out /path/to/run [--device cpu] [--scale 1.0]

Proves the pieces connect (loss decreases, the harness runs). Not a training run.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ..data.fixtures import fixture_texts, synthetic_records, write_jsonl  # noqa: E402
from ..data.records import read_records  # noqa: E402
from ..data.render import Renderer, render_record, span_examples  # noqa: E402
from ..eval.harness import run_harness  # noqa: E402
from ..model.heads import PortHeads  # noqa: E402
from ..model.lfm2_port import ControlTokens, PortBackbone, load_backbone  # noqa: E402
from .phases import smoke_phases  # noqa: E402
from .trainer import Trainer  # noqa: E402


def phase_curves(metrics_path: Path, window: int = 10) -> dict:
    by_phase = defaultdict(list)
    for line in metrics_path.read_text().splitlines():
        row = json.loads(line)
        by_phase[row["phase"]].append(row)
    curves = {}
    for phase, rows in by_phase.items():
        keys = [k for k in rows[0] if k not in {"step", "phase", "phase_step", "lr", "seconds"}
                and isinstance(rows[0][k], (int, float))]
        first, last = rows[:window], rows[-window:]
        mean = lambda rs, k: sum(r.get(k, 0.0) for r in rs) / len(rs)
        curves[phase] = {"steps": len(rows), "seconds_per_step": mean(rows, "seconds"),
                         **{k: {"first": mean(first, k), "last": mean(last, k)} for k in keys}}
    return curves


def main(argv=None):
    raise RuntimeError('Legacy A-F training is retired. Use the shared neuralese recipe and mandatory text warm-up; focused implementation diagnostics remain available under eval.')
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--cutoff", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=16)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    model, tokenizer = load_backbone(dtype=dtype, device="cpu")
    model.to(args.device)
    torch.manual_seed(0)
    backbone = PortBackbone(model, ControlTokens.from_tokenizer(tokenizer))
    heads = PortHeads(backbone, cutoff=args.cutoff, max_length=args.max_length).to(args.device)
    renderer = Renderer(tokenizer, backbone.controls)

    texts = fixture_texts(limit=200)
    spans_train = list(span_examples(renderer, texts[:160], 24, 8, 16, limit=400))
    spans_eval = list(span_examples(renderer, texts[160:], 24, 8, 16, limit=16))
    records_path = write_jsonl(synthetic_records(64), out / "fixture_port_records.jsonl")
    records_train = [render_record(renderer, r) for r in read_records([records_path], split="train")]
    records_eval = [render_record(renderer, r) for r in read_records([records_path], split="heldout")]
    cache_prefixes = [s.prefix for s in spans_eval[:4]]

    harness = lambda label: run_harness(
        backbone, heads, span_examples=spans_eval, rendered_records=records_eval,
        cache_prefixes=cache_prefixes, latency_prefix=spans_eval[0].prefix,
        out_path=out / f"harness_{label}.json", label=label)
    before = harness("before")
    trainer = Trainer(backbone, heads, smoke_phases(args.scale), out, span_train=spans_train,
                      records_train=records_train, checkpoint_every=10_000)
    trainer.run()
    after = harness("after")

    def brief(report):
        spans, records = report["spans"], report["records"]["families"]["fixture_projects"]
        return {"spans_nll": spans["nll"], "spans_correct_minus_shuffled": spans.get("correct_minus_shuffled"),
                "spans_gap_recovered": spans.get("gap_recovered"), "spans_stopping": spans["stopping"],
                "records_nll": records["nll"], "records_correct_minus_shuffled": records.get("correct_minus_shuffled"),
                "records_gap_recovered": records.get("gap_recovered"),
                "representation": spans["representation"], "cache_agreement": report["cache_agreement"],
                "temperature_sweep": report.get("temperature_sweep"),
                "latency_seconds": report["latency_seconds"]}

    summary = {"device": args.device, "cutoff": args.cutoff, "max_length": args.max_length,
               "train_spans": len(spans_train), "train_records": len(records_train),
               "eval_spans": len(spans_eval), "eval_records": len(records_eval),
               "wall_seconds": time.time() - started, "curves": phase_curves(out / "metrics.jsonl"),
               "before": brief(before), "after": brief(after)}
    (out / "smoke_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
