#!/usr/bin/env python3
"""Build deterministic, group-safe neutral curricula, preserving admission tiers."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_training_pipeline import atomic_json, digest_file


def records(paths):
    for path in paths:
        with Path(path).open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def hash_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def groups(row):
    return sorted(set([row.get("program_id") or row.get("group_id") or row["id"],
                       *(row.get("source_groups") or [])]))


def difficulty(row):
    code = row.get("completion") or json.dumps(row.get("target", {}))
    if any(x in code for x in ("installPackages", "import ", "import(", "fetch(", "await ")):
        return 3
    if any(x in code for x in ("reduce(", "sort(", "Map(", "Set(", "while ", "while(")):
        return 2
    if any(x in code for x in ("for ", "for(", "map(", "filter(", "if ", "if(")):
        return 1
    return 0


def prepare(output, code_paths=(), native_paths=(), teacher_paths=(), split_paths=(), registry=None, seed=42):
    output = Path(output)
    inputs = [*code_paths, *native_paths, *teacher_paths, *split_paths]
    identity = {"version": "natlang.training_curriculum/1", "seed": seed,
                "inputs": {str(Path(p).resolve()): digest_file(p) for p in inputs},
                "lanes": {"code": list(map(str, code_paths)), "native": list(map(str, native_paths)),
                          "teacher": list(map(str, teacher_paths)), "split": list(map(str, split_paths))},
                "registry": digest_file(registry) if registry else None,
                "builder_sha256": digest_file(__file__)}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["identity"] != identity:
            raise ValueError("curriculum inputs/settings changed; choose a new output directory")
        if any(digest_file(output / name) != value for name, value in old["outputs"].items()):
            raise ValueError("committed curriculum was modified")
        return old
    output.mkdir(parents=True, exist_ok=True)
    code, native, teacher = [list(records(p)) for p in (code_paths, native_paths, teacher_paths)]
    all_rows = [*code, *native, *teacher, *records(split_paths)]
    parent = {}
    def find(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]
    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    implementation = {}
    for row in all_rows:
        keys = groups(row)
        for key in keys:
            union(keys[0], key)
        fingerprint = row.get("implementation_sha256")
        if fingerprint:
            if fingerprint in implementation:
                union(keys[0], implementation[fingerprint])
            implementation[fingerprint] = keys[0]
    frozen = json.loads(Path(registry).read_text())["groups"] if registry else {}
    held, fixed = set(), {}
    for key, label in frozen.items():
        root = find(key)
        if root in fixed and fixed[root] != label:
            raise ValueError("new links connect conflicting frozen splits")
        fixed[root] = label
    for row in all_rows:
        root = find(groups(row)[0])
        if row.get("split") in ("test", "validation", "valid", "dev"):
            held.add(root)
    for root in held:
        if fixed.get(root) == "train":
            raise ValueError("new holdout overlaps previously frozen training group")
    roots = set(map(find, list(parent)))
    splits = {root: ("test" if root in held else fixed.get(root,
              "test" if int(hash_value([seed, root])[:8], 16) % 100 < 5 else "train")) for root in roots}
    # Explicit train is a preference, not permission to move a linked holdout into training.
    for row in all_rows:
        root = find(groups(row)[0])
        if row.get("split") == "train" and root not in held and root not in fixed:
            splits[root] = "train"
    def normalize(rows, lane):
        result, seen = [], set()
        for row in rows:
            if row.get("training_admission", {}).get("approved") is False:
                continue
            key = hash_value([row.get("prompt", row.get("messages")), row.get("completion", row.get("target"))])
            if key in seen:
                continue
            seen.add(key)
            root = find(groups(row)[0])
            result.append({**row, "program_id": root, "source_groups": sorted(set([root, *groups(row)])),
                           "split": splits[root], "curriculum_lane": lane, "difficulty": difficulty(row)})
        return sorted(result, key=lambda r: (r["difficulty"], hash_value([seed, r["id"]])))
    code = normalize(code, "general_code")
    native = normalize(native, "native_coding")
    teacher = normalize(teacher, "teacher_distillation")
    # API call examples belong in the coding curriculum, not foundational implementations.
    calls = [r for r in code if r.get("source", {}).get("name") == "xlam-function-calling-60k"]
    general = [r for r in code if r not in calls]
    # Modest bounded rehearsal, without multiplying tiny verified pools into fake breadth.
    rehearsal = sorted(general, key=lambda r: hash_value([seed, "rehearsal", r["id"]]))[:len(native + calls) // 4]
    coding = sorted([*native, *calls, *rehearsal], key=lambda r: (r["difficulty"], hash_value([seed, r["id"]])))
    outputs = {}
    counts = {}
    for name, rows in (("general", general), ("coding", coding), ("teacher", teacher)):
        target = output / f"{name}.jsonl"
        temporary = target.with_suffix(".pending")
        with temporary.open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            import os
            os.fsync(stream.fileno())
        temporary.replace(target)
        outputs[target.name] = digest_file(target)
        counts[name] = {"rows": len(rows), "splits": dict(Counter(r["split"] for r in rows)),
                        "difficulty": dict(Counter(r["difficulty"] for r in rows))}
    registry_out = output / "splits.json"
    atomic_json(registry_out, {"version": "natlang.split_registry/1", "groups": {key: splits[find(key)] for key in sorted(parent)}})
    outputs[registry_out.name] = digest_file(registry_out)
    manifest = {"identity": identity, "outputs": outputs, "counts": counts}
    atomic_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    for lane in ("code", "native", "teacher", "split-records"):
        parser.add_argument("--" + lane, nargs="*", default=[])
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.code, args.native, args.teacher, args.split_records, args.registry, args.seed)["counts"]))


if __name__ == "__main__":
    main()
