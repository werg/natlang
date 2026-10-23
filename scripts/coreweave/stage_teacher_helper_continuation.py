"""Prepare a versioned, inactive operator closure for helper continuation."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from scripts.coreweave import teacher_helper_continuation as custody


ROOT = Path("/home/werg/natlang")
DGXOPS = Path("/home/werg/data/dgxops")
SOURCE_DGXOPS = Path("/home/werg/data/dgxops")
API_SOURCE = Path("/home/werg/data/dgxops/teacher-zero-guided-control-closure-8c3358fa-20261022/scripts/coreweave/teacher_tranche_admission.py")
API_SOURCE_BINDING = {"path": str(API_SOURCE), "sha256": "33e4c92850f8fe05eeaffef958e33cba1d04235ec9c260fb2937d5a9ebc96ae1", "bytes": 61201}
INPUTS = {
    "selected-convert": {"path": DGXOPS / "teacher-zero-guided-control-closure-8c3358fa-20261022/scripts/coreweave/teacher_convert_operator.py", "sha256": "3c4cc6b5ddc5b22539a43ab644a05c9042610990c00094e5c905b00efef15c2b", "bytes": 9373, "kind": "converter", "tree": "selected/scripts/coreweave"},
    "selected-validate": {"path": DGXOPS / "teacher-zero-guided-control-closure-8c3358fa-20261022/scripts/coreweave/teacher_validate_operator.py", "sha256": "529e1dfe1335b12a00acb836c902d532b7d268126774f963957e45d0cae7d04b", "bytes": 19460, "kind": "validator", "tree": "selected/scripts/coreweave"},
    "flint-convert": {"path": DGXOPS / "flint-ready-native-admission-20261021/convert-flint-both-v2-0aad.py", "sha256": "a23e8200e89814a4d9d657f73e005fe54d0a235cbff53743c23956e1a29f7a06", "bytes": 6747, "kind": "converter", "tree": "flint"},
    "flint-validate": {"path": DGXOPS / "flint-ready-native-admission-20261021/strict-flint-both-v3-classified-siblings-0aad.py", "sha256": "6ee1c84689e2aa01aa94f3dc901fabafad23b3b466b4a9b7dcf8978f8c526cb1", "bytes": 15417, "kind": "validator", "tree": "flint"},
    "ready202-convert": {"path": DGXOPS / "teacher-ready202-admission-root-20261021/convert-current-teacher-both-v3-0aad.py", "sha256": "ce67c2ed8e018d7a1b0068434f5d9e2a943b5b5c8d2f5f2763e4013b1ec65a19", "bytes": 6792, "kind": "converter", "tree": "ready202"},
    "ready202-validate": {"path": DGXOPS / "teacher-ready202-admission-root-20261021/strict-current-teacher-both-v4-classified-siblings-0aad.py", "sha256": "ea8d1ccf08b2099f81742001d7d7685b0803e5de9a1d73ce5848369f7eeea364", "bytes": 15161, "kind": "validator", "tree": "ready202"},
    "drainer": {"path": DGXOPS / "teacher-zero-guided-control-closure-8c3358fa-20261022/drain-selected-and-current-teacher-data-v11-zero-eligible-active-repair-root-20261022.py", "sha256": "56bf718699b2a7b06497cc1c43b8b7bd2ac3c8b0f88f9bcee713e252d3b7417f", "bytes": 13396, "kind": "drainer", "tree": ""},
}


def file_binding(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"expected a regular source artifact: {target}")
    raw = target.read_bytes()
    return {"path": str(target.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _replace_ast_node(source: str, *, node_type: type, target_name: str, replacement: str) -> str:
    tree = ast.parse(source)
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, node_type):
            continue
        if node_type is ast.FunctionDef and node.name == target_name:
            matches.append(node)
        elif node_type is ast.Assign and any(isinstance(target, ast.Name) and target.id == target_name
                                             for target in node.targets):
            matches.append(node)
    if len(matches) != 1:
        raise ValueError(f"operator template needs exactly one {target_name} node, found {len(matches)}")
    node = matches[0]
    lines = source.splitlines(keepends=True)
    start, end = node.lineno - 1, node.end_lineno
    original = lines[start]
    indent = original[:len(original) - len(original.lstrip())]
    new_lines = [(indent + line if line else line) + "\n" for line in replacement.splitlines()]
    return "".join(lines[:start] + new_lines + lines[end:])


def _replace_string_constant_expression(source: str, value: str, replacement: str) -> str:
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    matches = [node for node in ast.walk(tree)
               if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value == value]
    if not matches:
        if value in source:
            raise ValueError("historical API digest is present outside a recognized string expression")
        return source
    edits = []
    for node in matches:
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[node.end_lineno - 1] + node.end_col_offset
        edits.append((start, end))
    for start, end in sorted(edits, reverse=True):
        source = source[:start] + replacement + source[end:]
    return source


def _exact_verifier_loader(helper_ref: dict[str, Any]) -> str:
    return "\n".join((
        "import hashlib as _continuation_hashlib",
        "import importlib.util as _continuation_importlib_util",
        "import pathlib as _continuation_pathlib",
        "import sys as _continuation_sys",
        f"_continuation_path = _continuation_pathlib.Path({helper_ref['path']!r})",
        f"_continuation_expected = {helper_ref!r}",
        "_continuation_raw = _continuation_path.read_bytes()",
        "if (str(_continuation_path.resolve()) != _continuation_expected['path'] or len(_continuation_raw) != _continuation_expected['bytes'] or _continuation_hashlib.sha256(_continuation_raw).hexdigest() != _continuation_expected['sha256']):",
        "    raise RuntimeError('pinned teacher helper continuation verifier changed')",
        "_continuation_spec = _continuation_importlib_util.spec_from_file_location('teacher_helper_continuation', _continuation_path)",
        "if _continuation_spec is None or _continuation_spec.loader is None:",
        "    raise RuntimeError('pinned teacher helper continuation verifier cannot be loaded')",
        "_continuation_module = _continuation_importlib_util.module_from_spec(_continuation_spec)",
        "_continuation_sys.modules['teacher_helper_continuation'] = _continuation_module",
        "_continuation_spec.loader.exec_module(_continuation_module)",
        "_helper_continuation = _continuation_module",
    ))


def transform_operator(source: str, *, kind: str, transition_ref: dict[str, Any], api_ref: dict[str, Any],
                       verifier_ref: dict[str, Any] | None = None) -> str:
    """Retarget one reviewed active script; source shape must be recognized exactly."""
    if not isinstance(verifier_ref, dict):
        raise ValueError("operator transformation requires the exact verifier-file binding")
    ast.parse(source)
    assignment = "\n".join((
        f"HELPER_OBSERVATION = _helper_continuation.LEGACY_OBSERVATION",
        f"HELPER_TRANSITION = {transition_ref!r}",
        f"SUCCESSOR_HELPER_BINDING = {api_ref!r}",
        f"SUCCESSOR_VERIFIER_BINDING = {verifier_ref!r}",
    ))
    result = _replace_ast_node(source, node_type=ast.Assign, target_name="HELPER_OBSERVATION", replacement=assignment)
    helper_body = "\n".join((
        "def verify_new_helpers():",
        "    _helper_continuation.verify_successor_bundle(",
        "        transition_binding=HELPER_TRANSITION,",
        "        successor_helper_binding=SUCCESSOR_HELPER_BINDING,",
        "        successor_verifier_binding=SUCCESSOR_VERIFIER_BINDING,",
        "        loaded_helper_path=ADMISSION_API_PATH, loaded_verifier_path=SUCCESSOR_VERIFIER_BINDING['path'], repository_root=ROOT)",
        "def verify_execution_helper(execution):",
        "    return _helper_continuation.verify_helper_continuation(",
        "        execution, transition_binding=HELPER_TRANSITION,",
        "        successor_helper_binding=SUCCESSOR_HELPER_BINDING,",
        "        successor_verifier_binding=SUCCESSOR_VERIFIER_BINDING,",
        "        loaded_helper_path=ADMISSION_API_PATH, loaded_verifier_path=SUCCESSOR_VERIFIER_BINDING['path'], repository_root=ROOT)",
    ))
    result = _replace_ast_node(result, node_type=ast.FunctionDef, target_name="verify_new_helpers", replacement=helper_body)
    if "ADMISSION_API_PATH =" not in result:
        old_import = "from scripts.coreweave.teacher_tranche_admission import"
        if result.count(old_import) < 1:
            raise ValueError("legacy operator no longer has a shared-helper import point")
        load_api = "\n".join((
            f"ADMISSION_API_PATH = pathlib.Path({api_ref['path']!r})",
            f"ADMISSION_API_BYTES = {api_ref['bytes']}",
            f"ADMISSION_API_SHA256 = {api_ref['sha256']!r}",
            "importlib.util",
            "_api_raw = ADMISSION_API_PATH.read_bytes()",
            "assert len(_api_raw) == ADMISSION_API_BYTES and hashlib.sha256(_api_raw).hexdigest() == ADMISSION_API_SHA256",
            "_api_spec = importlib.util.spec_from_file_location('scripts.coreweave.teacher_tranche_admission', ADMISSION_API_PATH)",
            "assert _api_spec is not None and _api_spec.loader is not None",
            "_api_module = importlib.util.module_from_spec(_api_spec)",
            "sys.modules['scripts.coreweave.teacher_tranche_admission'] = _api_module",
            "_api_spec.loader.exec_module(_api_module)",
            "import scripts.coreweave as _coreweave_package",
            "_coreweave_package.teacher_tranche_admission = _api_module",
            f"sys.path.insert(0, {str(Path(api_ref['path']).parent)!r})",
            _exact_verifier_loader(verifier_ref),
        ))
        result = result.replace(old_import, load_api + "\n" + old_import, 1)
        lines = result.splitlines(keepends=True)
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("import ") and "importlib.util" not in stripped:
                lines[index] = line.rstrip("\n") + ",importlib.util\n"
                break
        else:
            raise ValueError("legacy operator has no regular import line for the sibling loader")
        result = "".join(lines)
    else:
        # The selected closure already loads its sibling API under the correct module name.
        anchor = "_coreweave_package.teacher_tranche_admission = _api_module"
        if result.count(anchor) != 1:
            raise ValueError("selected operator API loader no longer has its expected package binding")
        anchor_at = result.index(anchor)
        line_start = result.rfind("\n", 0, anchor_at) + 1
        indent = result[line_start:anchor_at]
        if indent.strip():
            raise ValueError("selected operator API binding is no longer a standalone statement")
        loader = "\n".join(indent + line if line else line for line in _exact_verifier_loader(verifier_ref).splitlines())
        result = result.replace(anchor, anchor + "\n" + loader, 1)
    result = _replace_string_constant_expression(
        result, "e62b8d9ccaaf397d067706f15cc00417d08b4591aaa04542f273675dfe756054",
        "ADMISSION_API_SHA256")
    result = re.sub(r"execution\.get\(['\"]helper_continuation['\"]\)\s*==\s*HELPER_OBSERVATION",
                    "verify_execution_helper(execution) in {'historical', 'successor'}", result)
    helper_value = "HELPER_TRANSITION" if kind == "converter" else "execution.get('helper_continuation')"
    result = re.sub(r"(['\"]helper_continuation['\"]\s*:\s*)HELPER_OBSERVATION",
                    lambda match: match.group(1) + helper_value, result)
    if kind == "validator":
        result = result.replace("bound(ROOT/'scripts/coreweave/teacher_tranche_admission.py')", "bound(ADMISSION_API_PATH)")
        result = result.replace("bound(ROOT / 'scripts/coreweave/teacher_tranche_admission.py')", "bound(ADMISSION_API_PATH)")
    try:
        ast.parse(result)
    except SyntaxError as exc:
        lines = result.splitlines()
        lo, hi = max(1, exc.lineno - 3), min(len(lines), exc.lineno + 3)
        context = "\\n".join(f"{number}: {lines[number - 1]}" for number in range(lo, hi + 1))
        raise ValueError(f"transformed {kind} operator is invalid: {exc}; nearby source:\\n{context}") from exc
    if "HELPER_OBSERVATION = {'path'" in result or "HELPER_OBSERVATION={'path'" in result:
        raise ValueError("legacy observation literal remained in a transformed operator")
    return result


def _copy_exact(reference: dict[str, Any], destination: Path) -> bytes:
    source = reference["path"]
    actual = file_binding(source)
    if actual["sha256"] != reference["sha256"] or actual["bytes"] != reference["bytes"]:
        raise ValueError(f"active operator source changed since review: {source}")
    raw = Path(source).read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    return raw


def stage(destination: Path, review_report: Path) -> dict[str, Any]:
    destination = destination.resolve()
    if not destination.is_relative_to(DGXOPS) or destination.exists():
        raise ValueError("stage destination must be a new versioned directory under /home/werg/data/dgxops")
    review_input_ref = file_binding(review_report)
    review_raw = review_report.read_bytes()
    if file_binding(review_report) != review_input_ref:
        raise ValueError("independent review report changed while it was read")
    review = json.loads(review_raw)
    if (review.get("schema") != "natlang.teacher-helper-transition-review/1" or
            review.get("passed") is not True or review.get("full_check") is not True or
            review.get("source_commit") != custody.SUCCESSOR_COMMIT or
            review.get("training_admission") is not False or review.get("exposure") != 0):
        raise ValueError("independent review report is incomplete or not a non-admission proof")
    api_source_binding = file_binding(API_SOURCE)
    if api_source_binding != API_SOURCE_BINDING or review.get("successor_helper_sha256") != API_SOURCE_BINDING["sha256"]:
        raise ValueError("review report does not bind the exact staged successor helper")
    for name, spec in INPUTS.items():
        actual = file_binding(spec["path"])
        if actual["sha256"] != spec["sha256"] or actual["bytes"] != spec["bytes"]:
            raise ValueError(f"active operator source changed since review: {name}")
    destination.mkdir(parents=True)
    api_dest = destination / "selected/scripts/coreweave/teacher_tranche_admission.py"
    api_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(API_SOURCE, api_dest)
    api_ref = file_binding(api_dest)
    if api_ref["sha256"] != API_SOURCE_BINDING["sha256"] or api_ref["bytes"] != API_SOURCE_BINDING["bytes"]:
        raise ValueError("staged successor helper differs from the exact reviewed source")
    helper_dest = api_dest.with_name("teacher_helper_continuation.py")
    shutil.copyfile(Path(custody.__file__), helper_dest)
    verifier_ref = file_binding(helper_dest)
    review_dest = destination / "independent-review.json"
    review_dest.write_bytes(review_raw)

    closure_receipt, closure_transition = (Path(custody.SOURCE_CLOSURE_RECEIPT["path"]),
                                           Path(custody.SOURCE_CLOSURE_TRANSITION["path"]))
    receipt = json.loads(closure_receipt.read_bytes())
    transition = json.loads(closure_transition.read_bytes())
    if (file_binding(closure_receipt) != custody.SOURCE_CLOSURE_RECEIPT or
            file_binding(closure_transition) != custody.SOURCE_CLOSURE_TRANSITION or
            receipt.get("training_admission") is not False or
            receipt.get("code_transition") != {k: custody.SOURCE_CLOSURE_TRANSITION[k] for k in ("path", "sha256")} or
            transition.get("training_admission") is not False):
        raise ValueError("source-closure authorities differ from the exact historical transition")
    old = json.loads(Path(custody.LEGACY_OBSERVATION["path"]).read_bytes())
    implementation_files = {}
    for relative, expected in old["reviewed_external_helpers"]["files"].items():
        target = api_dest if relative == custody.LEGACY_API_RELATIVE else ROOT / relative
        actual = file_binding(target)
        if relative != custody.LEGACY_API_RELATIVE and any(actual[k] != expected[k] for k in ("sha256", "bytes")):
            raise ValueError(f"unreviewed helper implementation changed: {relative}")
        implementation_files[relative] = actual
    proof = {
        "schema": "natlang.teacher-helper-continuation-transition/1",
        "legacy_observation": custody.LEGACY_OBSERVATION,
        "legacy_authority": custody.LEGACY_AUTHORITY,
        "successor_helper": api_ref,
        "successor_verifier": verifier_ref,
        "implementation_files": implementation_files,
        "source_closure": {"receipt": file_binding(closure_receipt), "transition": file_binding(closure_transition),
                           "accounting": receipt["accounting"]},
        "review": file_binding(review_dest),
        "training_admission": False,
        "exposure": 0,
    }
    proof_path = destination / "teacher-helper-continuation-transition.json"
    proof_path.write_text(json.dumps(proof, sort_keys=True, indent=2) + "\n")
    transition_ref = file_binding(proof_path)
    custody.verify_successor_bundle(transition_binding=transition_ref, successor_helper_binding=api_ref,
                                    successor_verifier_binding=verifier_ref, loaded_helper_path=api_dest,
                                    loaded_verifier_path=helper_dest, repository_root=ROOT)

    output_refs = {}
    for key, spec in INPUTS.items():
        source = spec["path"]
        original = file_binding(source)
        if original["sha256"] != spec["sha256"] or original["bytes"] != spec["bytes"]:
            raise ValueError(f"active closure changed before staging: {source}")
        if spec["kind"] == "drainer":
            target = destination / "drain-selected-and-current-teacher-data.py"
            text = source.read_text()
            old_selected = str(SOURCE_DGXOPS / "teacher-zero-guided-control-closure-8c3358fa-20261022/scripts/coreweave")
            text = text.replace(f"selected_operators=Path('{old_selected}')", f"selected_operators=Path({str(api_dest.parent)!r})")
            replacements = {
                "convert='convert-flint-both-v2-0aad.py',strict='strict-flint-both-v3-classified-siblings-0aad.py'":
                f"convert={str(destination / 'flint/convert-flint-both-v2-0aad.py')!r},strict={str(destination / 'flint/strict-flint-both-v3-classified-siblings-0aad.py')!r}",
                "convert='convert-current-teacher-both-v3-0aad.py',strict='strict-current-teacher-both-v4-classified-siblings-0aad.py'":
                f"convert={str(destination / 'ready202/convert-current-teacher-both-v3-0aad.py')!r},strict={str(destination / 'ready202/strict-current-teacher-both-v4-classified-siblings-0aad.py')!r}",
            }
            for before, after in replacements.items():
                if text.count(before) != 1:
                    raise ValueError("drainer source selection is no longer the reviewed exact shape")
                text = text.replace(before, after)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        else:
            folder = destination / spec["tree"]
            api_path = api_dest
            helper_binding = {"path": str(api_path), "sha256": api_ref["sha256"], "bytes": api_ref["bytes"]}
            text = source.read_text()
            transformed = transform_operator(text, kind=spec["kind"], transition_ref=transition_ref,
                                             api_ref=helper_binding, verifier_ref=verifier_ref)
            if spec["tree"] == "selected/scripts/coreweave":
                target = folder / source.name
            else:
                target = folder / source.name
                folder.mkdir(parents=True, exist_ok=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(transformed)
        output_refs[key] = file_binding(target)
    # The selected API folder already contains the shared verifier; legacy operators
    # load that exact file by its pinned sibling path.
    manifest = {"schema": "natlang.teacher-helper-transition-operator-stage/1",
                "status": "prepared-inactive", "destination": str(destination),
                "prior_helper_observation": custody.LEGACY_OBSERVATION,
                "successor_transition": transition_ref, "successor_helper": api_ref,
                "successor_verifier": verifier_ref,
                "inputs": {key: {"path": str(value["path"]), "sha256": value["sha256"], "bytes": value["bytes"]}
                           for key, value in INPUTS.items()},
                "outputs": output_refs, "training_admission": False, "exposure": 0,
                "activation": "not performed; root must bind this prepared drainer and operator closure after review"}
    manifest_path = destination / "operator-transition.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return {"manifest": file_binding(manifest_path), "successor_transition": transition_ref,
            "successor_helper": api_ref, "successor_verifier": verifier_ref,
            "outputs": output_refs, "activation": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--review-report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.destination, args.review_report), sort_keys=True))


if __name__ == "__main__":
    main()
