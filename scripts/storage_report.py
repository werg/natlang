#!/usr/bin/env python3
"""Storage report: sizes per root, category and retention tier (plans/STORAGE_POLICY.md).

Scans the policy's roots with scripts/storage_retention.py (dry run, nothing deleted), adds `docker system df`,
~/.cache by top-level directory and the other top directories of the HDD root `data_hdd` (report only), and writes JSON +
Markdown. The HDD walk takes long: run it at the lowest best-effort I/O priority (idle class starves behind the archiver and syncs).

    ionice -c2 -n7 nice python3 scripts/storage_report.py [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import storage_retention as retention  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training' / 'neuralese'))
from natlang_neuralese.common.paths import root as machine_root  # noqa: E402

GIB = 2**30


def du(path: Path, timeout: int) -> int | None:
    try:
        out = subprocess.run(['du', '-sxb', str(path)], capture_output=True, text=True, timeout=timeout).stdout
        return int(out.split()[0]) if out else None
    except (subprocess.SubprocessError, ValueError, IndexError):
        return None


def docker_df() -> list[dict]:
    try:
        out = subprocess.run(['docker', 'system', 'df', '--format', '{{json .}}'], capture_output=True, text=True,
                             timeout=120).stdout
        return [json.loads(line) for line in out.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def extras(policy: dict, scanned: list[str], timeout: int) -> dict:
    report = {}
    cache = Path.home() / '.cache'
    report['cache'] = sorted(((child.name, du(child, timeout)) for child in cache.iterdir()
                              if child.is_dir() and not child.is_symlink()), key=lambda x: -(x[1] or 0)) \
        if cache.is_dir() else []
    external = machine_root('data_hdd')
    report['external_top'] = sorted(((str(child), du(child, timeout)) for child in external.iterdir()
                                     if child.is_dir() and not any(str(child) == s or s.startswith(str(child) + '/')
                                                                   for s in scanned)),
                                    key=lambda x: -(x[1] or 0)) if external.is_dir() else []
    report['disks'] = {str(p): dict(zip(('total', 'used', 'free'), shutil.disk_usage(p)))
                       for p in ('/', str(external)) if Path(p).exists()}
    report['docker'] = docker_df()
    return report


def markdown(report: dict, top: int = 25) -> str:
    gb = lambda b: f'{(b or 0) / GIB:,.1f}'
    lines = [f'# Storage report ({time.strftime("%Y-%m-%d %H:%M", time.localtime(report["time"]))})', '']
    lines += ['| Disk | Total GB | Used GB | Free GB |', '|---|---|---|---|']
    for disk, d in report['extras']['disks'].items():
        lines.append(f'| {disk} | {gb(d["total"])} | {gb(d["used"])} | {gb(d["free"])} |')
    lines += ['', '## By root and category', '', '| Root | Category | Files | GB |', '|---|---|---|---|']
    for root, cats in report['totals'].items():
        for cat, v in sorted(cats.items(), key=lambda kv: -kv[1]['bytes']):
            if v['bytes'] >= GIB // 10:
                lines.append(f'| {root} | {cat} | {v["files"]} | {gb(v["bytes"])} |')
    lines += ['', '## Retention tiers (files ≥ 1 GiB)', '', '| Tier | Files | GB |', '|---|---|---|']
    for tier, v in sorted(report['tiers'].items()):
        lines.append(f'| {tier} | {v["files"]} | {gb(v["bytes"])} |')
    for tier in ('PRUNE-AUTO', 'PRUNE-ASK', 'ACTIVE', 'KEEP'):
        items = sorted((e for e in report['entries'] if e['tier'] == tier), key=lambda e: -e['bytes'])[:top]
        if items:
            lines += ['', f'### {tier}: largest', '', '| GB | Path | Reason |', '|---|---|---|']
            lines += [f'| {gb(e["bytes"])} | `{e["path"]}` | {e["reason"]} |' for e in items]
    revisions = report.get('sync_revisions', {})
    lines += ['', f'## Sync revisions (rsync --backup copies): {len(revisions)} directories, '
                  f'{gb(sum(revisions.values()))} GB', '', '| Directory | GB |', '|---|---|']
    lines += [f'| `{d}` | {gb(b)} |' for d, b in sorted(revisions.items(), key=lambda kv: -kv[1])[:15]]
    lines += ['', '## ~/.cache', '', '| Dir | GB |', '|---|---|']
    lines += [f'| {name} | {gb(size)} |' for name, size in report['extras']['cache'][:15]]
    lines += ['', f"## Other {machine_root('data_hdd')} top directories (report only)", '', '| Dir | GB |', '|---|---|']
    lines += [f'| {name} | {gb(size)} |' for name, size in report['extras']['external_top']]
    lines += ['', '## Docker', '', '| Type | Total | Size | Reclaimable |', '|---|---|---|---|']
    lines += [f'| {d.get("Type")} | {d.get("TotalCount")} | {d.get("Size")} | {d.get("Reclaimable")} |'
              for d in report['extras']['docker']]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default=str(retention.STATE))
    p.add_argument('--policy', default=str(retention.POLICY))
    p.add_argument('--du-timeout', type=int, default=7200)
    a = p.parse_args(argv)
    args = argparse.Namespace(apply=False, policy=a.policy, roots=None, approvals=str(retention.APPROVALS),
                              manifest=str(retention.STATE / 'deletions.jsonl'), no_docker=True)
    report = retention.run(args)
    policy = retention.load_policy(a.policy)
    report['extras'] = extras(policy, policy['roots']['scan'], a.du_timeout)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%dT%H%M')
    (out / f'report-{stamp}.json').write_text(json.dumps(report, indent=1))
    (out / f'report-{stamp}.md').write_text(markdown(report))
    (out / 'report-latest.md').write_text(markdown(report))
    print(str(out / f'report-{stamp}.md'))


if __name__ == '__main__':
    main()
