from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from neuralese_authored_worlds import iterate_world, reducer_world
from neuralese_source_world_builder import build_proof, validate_world_rows

REV = "authored-semantic-source-worlds-v15/12"
OUT = ROOT / "runs/neuralese-successor-v15-prepared-20261007-candidate-v12"
SPECS_PATH = ROOT / "scripts/neuralese_successor_v15_source_specs.json"
SPECS = json.loads(SPECS_PATH.read_text())
if SPECS.get("revision") != REV:
    raise ValueError(f"source spec revision {SPECS.get('revision')!r} does not match builder {REV!r}")
NESTED_SPECS = SPECS["nested"]
ITERATE_SPECS = SPECS["iterate"]
IDENTITY_EVIDENCE = SPECS["identity_evidence"]
AUTHORITY_EVIDENCE = SPECS["authority_evidence"]

# Categories are assigned from the actual packet object/site facts. This table
# is reviewed source classification, independent of acceptance labels.
TEAM_ORDER = {
    "archive_sound_rights": ["oral_history", "field_recording", "oral_history", "field_recording", "field_recording", "oral_history"],
    "transit_accessibility_repair": ["platform", "vehicle", "platform", "vehicle", "platform", "vehicle"],
    "volunteer_cold_chain": ["cultures", "produce", "cultures", "produce", "cultures", "produce"],
    "habitat_field_permits": ["wetland", "cliff", "wetland", "cliff", "wetland", "cliff"],
    "translation_editorial_clearance": ["Kalo", "Ruma", "Kalo", "Ruma", "Kalo", "Ruma"],
    "craft_repair_material_release": ["woodshop", "textile", "woodshop", "textile", "textile", "woodshop"],
    "library_oral_history_release": ["web_audio", "exhibit", "web_audio", "exhibit", "web_audio", "exhibit"],
    "watershed_sampling_safety": ["upper_basin", "lower_basin", "upper_basin", "lower_basin", "upper_basin", "lower_basin"],
    "community_energy_match": ["heat_pump", "solar", "heat_pump", "solar", "heat_pump", "solar"],
    "school_accessible_trip": ["wheelchair_route", "sensory_plan", "wheelchair_route", "sensory_plan", "wheelchair_route", "sensory_plan"],
    "clinic_interpreter_roster": ["Kalo", "Ruma", "Kalo", "Ruma", "Kalo", "Ruma"],
    "neighborhood_tree_work": ["nesting_tree", "street_tree", "nesting_tree", "street_tree", "nesting_tree", "street_tree"],
}

# Repair category/object contradictions in the preserved source specifications.
for spec in NESTED_SPECS:
    if spec["slug"] == "transit_accessibility_repair":
        spec["objects"][5], spec["sites"][5] = "vehicle ramp latch repair", "vehicle V-11"
    elif spec["slug"] == "craft_repair_material_release":
        spec["objects"] = ["spindle wheel", "loom shuttle", "oak stool leg", "embroidered apron panel", "apron seam", "maple mallet"]
        spec["sites"] = ["bench W-2", "loom T-1", "bench W-4", "dye table T-2", "dye table T-3", "bench W-9"]
    elif spec["slug"] == "school_accessible_trip":
        spec["objects"] = ["step-free museum trip", "quiet-room garden trip", "step-free aquarium trip", "quiet-room history walk", "step-free planetarium trip", "quiet-room library trip"]
    elif spec["slug"] == "habitat_field_permits":
        spec["sites"][4] = "Juniper Marsh"
        spec["objects"][5] = "cliff-edge tidepool census"
        spec["sites"][5] = "cliff Cove 6"


def _balanced_accepts(team_order: list[str], split_index: int) -> set[int]:
    groups = {team: [i for i, assigned in enumerate(team_order) if assigned == team]
              for team in dict.fromkeys(team_order)}
    selected: set[int] = set()
    group_counts: dict[str, int] = {}
    for team_index, (team, indices) in enumerate(groups.items()):
        count = max(1, len(indices) // 2)
        group_counts[team] = count
        rotate = (split_index + team_index) % len(indices)
        rotated = indices[rotate:] + indices[:rotate]
        selected.update(rotated[:count])
    while len(selected) < 3:
        preferred = list(groups)[split_index % len(groups)]
        candidates = [i for i in groups[preferred] if i not in selected]
        if not candidates:
            preferred = next(team for team in groups if any(i not in selected for i in groups[team]))
            candidates = [i for i in groups[preferred] if i not in selected]
        selected.add(candidates[(split_index // 2) % len(candidates)])
    if len(selected) != 3:
        raise AssertionError("nested category balancing must yield exactly three positives")
    for indices in groups.values():
        if not (0 < len(selected.intersection(indices)) < len(indices)):
            raise AssertionError("every nested folder must contain positive and negative records")
    return selected


def _validate_role_metadata(spec: dict, team_order: list[str]) -> None:
    """Reject semantic templates that would substitute an organization for a person."""
    for team_name, team in spec["teams"].items():
        uses_person = any("{person}" in team.get(key, "") for key in ("good", "bad"))
        source = team.get("person_source")
        if uses_person and source not in {"identity", "role_actor"}:
            raise ValueError(f"{spec['slug']}/{team_name}: person template needs person_source")
        if not uses_person and source:
            raise ValueError(f"{spec['slug']}/{team_name}: unused person_source {source!r}")
        if source == "role_actor":
            names = team.get("role_actor_names")
            if not isinstance(names, list) or len(names) != len(team_order):
                raise ValueError(f"{spec['slug']}/{team_name}: role_actor_names must align with all item indices")
            if not team.get("actor_role"):
                raise ValueError(f"{spec['slug']}/{team_name}: role_actor requires an actor_role label")
            for index, assigned_team in enumerate(team_order):
                if assigned_team == team_name and not names[index]:
                    raise ValueError(f"{spec['slug']} index {index}: missing {team['actor_role']} name")
                if assigned_team == team_name and names[index] == spec["identities"][index]:
                    raise ValueError(f"{spec['slug']} index {index}: role actor duplicates the organization/item identity")
        elif source == "identity":
            if not spec.get("identities") or len(spec["identities"]) != len(team_order):
                raise ValueError(f"{spec['slug']}/{team_name}: identity source must align with item indices")


def make_nested_rows():
    rows = []
    split_seen = {"train": 0, "test": 0}
    for spec in NESTED_SPECS:
        split_index = split_seen[spec["split"]]
        split_seen[spec["split"]] += 1
        team_order = TEAM_ORDER[spec["slug"]]
        _validate_role_metadata(spec, team_order)
        by_team = {team: [i for i, name in enumerate(team_order) if name == team]
                   for team in dict.fromkeys(team_order)}
        accepted_indices = _balanced_accepts(team_order, split_index)
        records = []
        for index in range(6):
            item_id = f"{spec['slug'][:3].upper()}-{201 + index * 7:03d}"
            team_name = team_order[index]
            team = spec["teams"][team_name]
            identity = spec["identities"][index]
            role_person = team.get("role_actor_names", [None] * len(team_order))[index]
            person_source = team.get("person_source")
            if person_source == "role_actor":
                if not role_person:
                    raise ValueError(f"{spec['slug']} {item_id}: missing declared {team['actor_role']}")
                local_person = role_person
            elif person_source == "identity":
                local_person = identity
            else:
                local_person = identity
            authority, thing = spec["authorities"][index], spec["objects"][index]
            site, revision = spec["sites"][index], spec["revisions"][index]
            accepted = index in accepted_indices
            if accepted:
                fail_clause = None
            else:
                negatives = [i for i in by_team[team_name] if i not in accepted_indices]
                fail_clause = (negatives.index(index) + split_index + list(by_team).index(team_name)) % 3
            supports = [fail_clause != clause for clause in range(3)]
            identity_template = IDENTITY_EVIDENCE[spec["slug"]][0 if supports[0] else 1]
            identity_sentence = identity_template.format(person=identity, thing=thing, item_id=item_id,
                                                         site=site, revision=revision)
            authority_template = AUTHORITY_EVIDENCE[spec["slug"]][0 if supports[1] else 1]
            authority_sentence = authority_template.format(authority=authority, revision=revision, item_id=item_id)
            local_sentence = (team["good"] if supports[2] else team["bad"]).format(
                n=item_id, person=local_person, thing=thing, site=site, rev=revision, vehicle=site)
            record_qualifier = team.get("record_qualifier", "")
            qualified_site = f"{site} {record_qualifier}".strip()
            records.append({"id": item_id, "team": team_name, "thing": thing, "site": site,
                "identity": identity, "role_actor": role_person, "role": team.get("actor_role"),
                "text": f"{item_id} concerns {thing} at {qualified_site}. Classification: {team_name.replace('_', ' ')}. {identity_sentence} {authority_sentence} {local_sentence}",
                "authority_evidence": authority_sentence,
                "supports": supports, "failed_clause": fail_clause, "accepted": accepted})
        row = reducer_world(spec["slug"], spec["split"], spec["domain"], spec["inherited"],
            {key: value["rule"] for key, value in spec["teams"].items()}, records,
            spec["clauses"], "The complete matching item FileHandle is the only source for that item's facts.", revision=REV)
        audit_by_id = {item["id"]: item for item in row["_audit"]["records"]}
        for item in records:
            audit_by_id[item["id"]].update({"identity": item["identity"],
                "role_actor": item["role_actor"], "role": item["role"]})
        rows.append(row)
    return rows


def make_iterate_rows():
    return [iterate_world(spec["slug"], spec["split"], spec["domain"], spec["fields"], spec["passes"],
                          spec["initial"], spec["packet"], revision=REV) for spec in ITERATE_SPECS]


def build_rows():
    rows = make_nested_rows() + make_iterate_rows()
    rows.sort(key=lambda row: row["curriculum"]["shape"])
    return rows


def _atomic_write(path: Path, data: bytes):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(output_dir: Path | None = None):
    output_dir = output_dir or OUT
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = build_rows()
    audits = {row["id"]: row.pop("_audit") for row in rows}
    stats = validate_world_rows(rows)
    raw = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows).encode()
    _atomic_write(output_dir / "source.cases.jsonl", raw)
    proof = build_proof(rows, raw)
    proof.update({"world_audits": audits,
        "nested_reducers": {row["curriculum"]["shape"]: audits[row["id"]] for row in rows if "nested-reducer" in row["curriculum"]["shape"]},
        "iterate_passes": {row["curriculum"]["shape"]: audits[row["id"]]["passes"] for row in rows if "-iterate" in row["curriculum"]["shape"]}})
    _atomic_write(output_dir / "scripted-source-proof.json", (json.dumps(proof, indent=2, ensure_ascii=False) + "\n").encode())
    nested_audits = [audit for audit in audits.values() if "records" in audit]
    iter_audits = [audit for audit in audits.values() if "passes" in audit]
    rank_counts = {split: [0] * 6 for split in ("train", "test")}
    category_counts = {}
    for row in rows:
        if "nested-reducer" not in row["curriculum"]["shape"]:
            continue
        audit = audits[row["id"]]
        group_counts = {}
        for item in audit["records"]:
            team = item["team"]
            counts = group_counts.setdefault(team, [0, 0])
            counts[0 if item["accepted"] else 1] += 1
        category_counts[row["curriculum"]["shape"]] = group_counts
        for rank, item in enumerate(sorted(audit["records"], key=lambda record: record["id"])):
            rank_counts[row["split"]][rank] += int(item["accepted"])
    quality = {"schema":"natlang.neuralese-source-quality-review/1", "revision":REV,
        "source_cases_sha256":hashlib.sha256(raw).hexdigest(),
        "review_type":"deterministic source consistency; independent semantic review remains required",
        "world_count":stats["world_count"],"train_count":stats["train_count"],"test_count":stats["test_count"],
        "nested_reducer_worlds":len(nested_audits),"iterate_worlds":len(iter_audits),
        "nested_records":sum(len(audit["records"]) for audit in nested_audits),
        "positive_records":sum(sum(item["accepted"] for item in audit["records"]) for audit in nested_audits),
        "negative_records":sum(sum(not item["accepted"] for item in audit["records"]) for audit in nested_audits),
        "iterate_pass_count_distribution":{str(count):sum(len(audit["passes"])==count for audit in iter_audits) for count in (4,5,6)},
        "positive_lexical_ranks_per_split":rank_counts,"positive_negative_counts_by_folder":category_counts,
        "checks":{"unique_factual_groups":True,"split_groups_disjoint":True,
            "every_nested_negative_fails_exactly_one_clause":True,"every_nested_folder_has_mixed_labels":True,
            "category_assignments_follow_packet_facts":True,"full_draft_carried_each_iterate_pass":True,
            "each_pass_has_source_evidence":True,"model_calls":0,"provider_calls":0,"teacher_trajectories":0,
            "admission_granted":False},
        "limits":["Scripted values prove source/runtime plumbing only; they are not teacher observations.",
            "Independent semantic review is still required."]}
    _atomic_write(output_dir / "source-quality-review.json", (json.dumps(quality,indent=2,ensure_ascii=False)+"\n").encode())
    manifest={"schema":"natlang.neuralese-semantic-source-v15/1","campaign":"successor-v15-prepared-20261007-candidate-v12",
        "task_count":stats["world_count"],"train_count":stats["train_count"],"test_count":stats["test_count"],"source_revision":REV,
        "source_cases":"source.cases.jsonl","source_cases_sha256":hashlib.sha256(raw).hexdigest(),
        "source_quality_review":"source-quality-review.json","scripted_proof":"scripted-source-proof.json",
        "runtime_proof":"runtime-reference-proof.json","runtime_audit":"source-action-review.jsonl",
        "builder":"scripts/build_neuralese_successor_v15_candidate.py","builder_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "specs":"scripts/neuralese_successor_v15_source_specs.json","specs_sha256":hashlib.sha256(SPECS_PATH.read_bytes()).hexdigest(),
        "admission":"candidate only; no teacher generation, corpus admission, or training launch",
        "worlds":[row["curriculum"]["shape"] for row in rows]}
    _atomic_write(output_dir/"source-manifest.json",(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n").encode())
    print(json.dumps({**stats,"source_cases_sha256":hashlib.sha256(raw).hexdigest(),
        "nested_worlds":len(nested_audits),"iterate_worlds":len(iter_audits),"positive_ranks_per_split":rank_counts,
        "category_counts":category_counts,"worlds":manifest["worlds"]},indent=2))


if __name__ == "__main__":
    main()
