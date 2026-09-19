#!/usr/bin/env python3
"""Measure write-confidence discrimination on known semantic labels.

Uses familiar generated templates with fresh instances; this is a diagnostic,
not proof of calibration on novel programs. Type failures stay a separate class.
"""
import argparse
import json
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.generate_failures import matched_cases
from scripts.probe_parallel import completed_cases, total_usage
from natlang.native import NativeCallDecoder
from natlang.decoder import ChatTurn
from natlang.runtime import Runtime, Session
from natlang.types import TypeEnv
from natlang.tool_agent import ToolAgent
from natlang.values import load_program, dump


class InjectFirst:
    def __init__(self, inner, fault):
        self.inner, self.fault = inner, fault
    @property
    def deadline(self):
        return self.inner.deadline
    @deadline.setter
    def deadline(self, v):
        self.inner.deadline = v
    def review(self, *args, **kw):
        return self.inner.review(*args, **kw)
    def chat(self, *args, **kw):
        if self.fault:
            f, self.fault = self.fault, None
            return ChatTurn([(f[0], f[1])], completion_tokens=1)
        return self.inner.chat(*args, **kw)


def label_proposal(c, call):
    name, args = call
    if name != 'write' or args.get('path') != 'return' or 'value' not in args:
        return 'unscored'
    root = load_program(c['root'])
    session = Session(Runtime(None), root, TypeEnv())
    result = session.apply(name, args)
    if result.kind != 'ok' or not session.finish():
        return 'validation_failure'
    if c['failure'] or dump(root.ret) != c['expected']:
        return 'semantic_mismatch'
    return 'correct'


def summarize(rows):
    scored = [v for r in rows for v in r['values'] if v['label'] in ('correct', 'semantic_mismatch') and v['confidence'] is not None]
    bins = []
    for lo, hi in ((0, .2), (.2, .5), (.5, .8), (.8, .95), (.95, 1.000001)):
        bucket = [v for v in scored if lo <= v['confidence'] < hi]
        bins.append({'lower': lo, 'upper': min(1, hi), 'n': len(bucket),
                     'semantic_mismatches': sum(v['label'] == 'semantic_mismatch' for v in bucket),
                     'observed_correct_fraction': sum(v['label'] == 'correct' for v in bucket) / len(bucket) if bucket else None,
                     'mean_raw_score': sum(v['confidence'] for v in bucket) / len(bucket) if bucket else None})
    thresholds = []
    good = [v for v in scored if v['label'] == 'correct']
    bad = [v for v in scored if v['label'] == 'semantic_mismatch']
    for threshold in (.2, .5, .8, .95):
        thresholds.append({'threshold': threshold, 'flagged_bad': sum(v['confidence'] < threshold for v in bad),
                           'total_bad': len(bad), 'flagged_good': sum(v['confidence'] < threshold for v in good),
                           'total_good': len(good)})
    return {'cases': len(rows), 'correct_cases': sum(r['correct'] for r in rows),
            'scored_values': len(scored), 'unknown_confidence': sum(v['confidence'] is None for r in rows for v in r['values']),
            'validation_failures': sum(v['label'] == 'validation_failure' for r in rows for v in r['values']),
            'bins': bins, 'thresholds': thresholds}



def calibrate_threshold(rows):
    """Fit a review trigger on even groups; report discrimination on odd groups.

    This calibrates a decision threshold, not a probability of correctness.
    Distinct instances share templates, so transfer still needs a novel-template probe.
    """
    def points(parity):
        return [v for r in rows if r["group"] % 2 == parity for v in r["values"]
                if v["label"] in ("correct", "semantic_mismatch") and v["confidence"] is not None]
    train, test = points(0), points(1)
    def rates(items, threshold):
        good = [v for v in items if v["label"] == "correct"]
        bad = [v for v in items if v["label"] == "semantic_mismatch"]
        tp = sum(v["confidence"] < threshold for v in bad)
        fp = sum(v["confidence"] < threshold for v in good)
        return {"bad": len(bad), "good": len(good), "flagged_bad": tp, "flagged_good": fp,
                "balanced_accuracy": (tp / len(bad) + 1 - fp / len(good)) / 2 if bad and good else None}
    if not train or len({v["label"] for v in train}) < 2:
        return {"threshold": None, "reason": "Both classes required in calibration partition"}
    values = sorted({v["confidence"] for v in train})
    candidates = [0.] + [(a+b)/2 for a,b in zip(values, values[1:])] + [1.]
    best = max(candidates, key=lambda t: (rates(train, t)["balanced_accuracy"], -t))
    return {"threshold": best, "objective": "balanced accuracy, ties prefer fewer reviews",
            "calibration_even_groups": rates(train, best), "held_out_odd_groups": rates(test, best)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--server', default='http://127.0.0.1:8080')
    ap.add_argument('--groups', type=int, default=10)
    ap.add_argument('--seed', type=int, default=992)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--careful-threshold', type=float)
    ap.add_argument('--inject-fault', action='store_true', help='shared first rejected call; caller policy will stop before model on those cases')
    ap.add_argument("--partition", choices=["all", "calibration", "test"], default="all")
    args = ap.parse_args()
    if min(args.groups, args.workers) < 1:
        ap.error('groups and workers must be positive')
    jobs = [(g, c) for g in range(args.groups) if args.partition == "all" or g % 2 == (args.partition == "test")
            for c in matched_cases(args.seed, g)]
    prompt = (ROOT / 'natlang/prompts/tools_delegate.md').read_text()
    def run(job):
        group, c = job
        dec = NativeCallDecoder(args.server)
        wrapper = InjectFirst(dec, c['fault']) if args.inject_fault else dec
        proposals, reviews, log = [], [], []
        rt = Runtime(lambda lam: ToolAgent(wrapper, system_prompt=prompt, temperature=0, max_turns=16,
            max_tokens=2400, max_seconds=90, careful_threshold=args.careful_threshold,
            proposals=proposals, reviews=reviews, log=log))
        out, val = rt.run_root(load_program(c['root']))
        if c['failure']:
            correct = out.kind == 'quiesced' and (out.detail.startswith(c['failure'] + ':') or out.detail.startswith('validation failed:') or out.detail.startswith('careful review ' + ('error:' if c['failure'] == 'error' else 'blocker:')))
        else:
            correct = out.kind == 'done' and dump(val) == c['expected']
        values = []
        for proposal in proposals:
            for i, call in enumerate(proposal['calls']):
                if call[0] != 'write':
                    continue
                confidence = proposal['value_confidence'][i] if i < len(proposal['value_confidence']) else None
                values.append({'call': call, 'label': label_proposal(c, call),
                               'confidence': confidence['geometric_mean'] if confidence else None,
                               'metrics': confidence, 'released': proposal['released']})
        return {'group': group, 'case': c['name'], 'status': out.kind, 'detail': out.detail,
                'correct': correct and rt.emitted == c['effects'], 'values': values,
                'reviews': reviews, 'log': log, 'usage': dec.usage}
    rows, started = [], time.monotonic()
    for index, row in completed_cases(run, jobs, args.workers):
        rows.append({'index': index, **row})
        rows.sort(key=lambda r: r['index'])
        doc = {'args': {k: str(v) for k, v in vars(args).items()}, 'system_prompt': prompt,
               'wall_seconds': time.monotonic() - started, 'rows': rows, 'usage': total_usage(rows), 'summary': summarize(rows), 'threshold_calibration': calibrate_threshold(rows)}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_suffix('.tmp')
        tmp.write_text(json.dumps(doc, indent=2) + '\n')
        tmp.replace(args.out)
        print(f'{len(rows)}/{len(jobs)} {row["case"]}: {row["status"]} correct={row["correct"]}', flush=True)


if __name__ == '__main__':
    main()
