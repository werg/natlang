import csv
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_translation_source_tasks.py"
SPEC = importlib.util.spec_from_file_location("translation_source_tasks", SCRIPT)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def fixture_repo(root: Path, *, n_train: int = 20, n_valid: int = 5, n_test: int = 7) -> Path:
    for rel in mod.PINNED_FILES:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".csv"):
            split = Path(rel).stem
            n = {"train": n_train, "valid": n_valid, "test": n_test}[split]
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["", *mod.LANGUAGES])
                for i in range(n):
                    values = [str(i)]
                    values.extend(f"{name} sentence {i} café — naïve" for name in mod.LANGUAGES)
                    writer.writerow(values)
        else:
            path.write_text(f"pinned fixture {rel}\n", encoding="utf-8")
    return root


def test_parallel_sentence_role_is_shared_across_directions_and_references_are_separate(tmp_path):
    repo = fixture_repo(tmp_path / "repo")
    pairs = [("english", "indonesian"), ("indonesian", "english"), ("javanese", "indonesian")]
    tasks, refs, manifest = mod.build_translation_tasks(repo, pairs=pairs, max_train_rows=15, max_valid_rows=3, verify_pin=False)
    assert len(tasks) == len(refs)
    assert {t["role"] for t in tasks if t["original_split"] == "valid"} == {"validation"}
    roles_by_group = {}
    for task in tasks:
        roles_by_group.setdefault(task["source_group"], set()).add(task["role"])
    assert all(len(roles) == 1 for roles in roles_by_group.values())
    # Any one selected source sentence appears in all requested directions under one stable role.
    train_groups = [g for g in roles_by_group if "/train/" in g]
    assert train_groups
    for group in train_groups:
        rows = [t for t in tasks if t["source_group"] == group]
        assert {t["source_language"] + "->" + t["target_language"] for t in rows} == {"english->indonesian", "indonesian->english", "javanese->indonesian"}
    assert all("reference_target" not in task for task in tasks)
    assert all(ref["visibility"] == "host-only" for ref in refs)
    assert all(task["candidate_status"] == "held_semantic_evaluator_pending" for task in tasks)
    assert manifest["semantic_evaluator"]["exact_reference_string_grading"] is False
    assert manifest["protected_original_test"]["exported_as_tasks"] is False
    assert manifest["protected_original_test"]["row_count"] == 7
    assert all("café" in task["input"]["text"] for task in tasks)


def test_direction_selection_caps_are_deterministic_and_keep_original_splits(tmp_path):
    repo = fixture_repo(tmp_path / "repo")
    pair = [("english", "indonesian")]
    first = mod.build_translation_tasks(repo, pairs=pair, max_train_rows=8, max_valid_rows=2, verify_pin=False)
    second = mod.build_translation_tasks(repo, pairs=pair, max_train_rows=8, max_valid_rows=2, verify_pin=False)
    assert first == second
    tasks, _refs, manifest = first
    assert len({t["source_group"] for t in tasks if t["original_split"] == "train"}) == 8
    assert len({t["source_group"] for t in tasks if t["original_split"] == "valid"}) == 2
    assert manifest["selected_parallel_rows"] == {"train": 8, "valid": 2}
    assert {t["original_split"] for t in tasks} == {"train", "valid"}


def test_all_directed_language_pairs_are_selectable_but_self_translation_is_rejected():
    assert len(mod.parse_pairs("all")) == 12 * 11
    assert ("english", "acehnese") in mod.parse_pairs("en->acehnese")
    assert ("acehnese", "english") in mod.parse_pairs("acehnese->en")
    assert len(mod.DEFAULT_PAIRS) == 42
    try:
        mod.parse_pairs("english->english")
    except ValueError:
        pass
    else:
        raise AssertionError("self translation must not be accepted")


def test_manifest_distinguishes_dataset_and_repository_code_licenses(tmp_path):
    repo = fixture_repo(tmp_path / "repo")
    _tasks, _references, manifest = mod.build_translation_tasks(repo, pairs=[("english", "indonesian")], max_train_rows=2, max_valid_rows=0, verify_pin=False)
    assert manifest["dataset_license"] == "CC-BY-SA-4.0"
    assert manifest["license_evidence"]["software_license"] == "Apache-2.0 (repository code only; not applied to the dataset)"
    assert manifest["rights_review"]["item_level_upstream_rights_review"] == "pending"
