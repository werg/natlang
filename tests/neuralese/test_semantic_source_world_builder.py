import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "scripts/build_neuralese_successor_v14_candidate.py"
SPEC = importlib.util.spec_from_file_location("v14_source_builder", BUILDER)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)

V15_SPEC = importlib.util.spec_from_file_location(
    "v15_source_builder", ROOT / "scripts/build_neuralese_successor_v15_candidate.py")
v15 = importlib.util.module_from_spec(V15_SPEC)
V15_SPEC.loader.exec_module(v15)


def test_v15_source_facts_explicitly_cover_local_rule_qualifiers():
    rows = v15.make_nested_rows()
    watershed = next(row for row in rows if 'watershed_sampling_safety' in row['id'])
    for path, text in watershed['semantics']['folder_files'].items():
        if '/upper_basin/items/' in path:
            assert 'Steep-bank tether check' in text and 'at steep upper bank' in text
    clinic = next(row for row in rows if 'clinic_interpreter_roster' in row['id'])
    for path, text in clinic['semantics']['folder_files'].items():
        if path.endswith('/override.json'):
            assert 'C1 above B2 above B1' in text
        if '/items/' in path:
            assert 'current CEFR' in text and 'confidentiality training' in text
            assert 'through' in text

    community = next(row for row in rows if 'community_energy_match' in row['id'])
    for path, text in community['semantics']['folder_files'].items():
        if '/solar/items/' in path:
            assert 'Roof engineer ' in text and 'signed roof capacity note' in text
    habitat = next(row for row in rows if 'habitat_field_permits' in row['id'])
    for path, text in habitat['semantics']['folder_files'].items():
        if '/cliff/items/' in path:
            assert 'Local ranger ' in text and 'assigned to ' in text
    trees = next(row for row in rows if 'neighborhood_tree_work' in row['id'])
    for path, text in trees['semantics']['folder_files'].items():
        if '/nesting_tree/items/' in path:
            assert 'with a confirmed active nest' in text
            assert 'active nest at' in text
    volunteer = next(row for row in rows if 'volunteer_cold_chain' in row['id'])
    for path, text in volunteer['semantics']['folder_files'].items():
        if '/cultures/items/' in path:
            assert 'live ' in text and ('sealed chilled transport' in text or 'ambient transport' in text)
    for row in rows:
        for path, text in row['semantics']['folder_files'].items():
            if path in row['semantics']['expected_files'] and path.endswith('.md'):
                assert row['semantics']['expected_files'][path] == text


def test_v15_role_people_are_separate_from_team_and_item_identities():
    rows = v15.make_nested_rows()
    expected = {
        ('archive_sound_rights', 'ARC-201'): ('Mara Velin', 'Tern Street Radio'),
        ('community_energy_match', 'COM-208'): ('Nia Bell', 'Hill Lantern Co-op'),
        ('habitat_field_permits', 'HAB-222'): ('D. Vale', 'Salt Grass Watch'),
        ('neighborhood_tree_work', 'NEI-215'): ('L. Arden', 'Willow Steps Council'),
        ('school_accessible_trip', 'SCH-222'): ('L. Ames', 'Class 5D'),
        ('translation_editorial_clearance', 'TRA-236'): ('V. Neri', 'Ruma food-safety card edition 6'),
    }
    for world, item_id in expected:
        row = next(row for row in rows if world in row['id'])
        item = next(record for record in row['_audit']['records'] if record['id'] == item_id)
        actor, identity = expected[(world, item_id)]
        assert item['role_actor'] == actor
        assert item['identity'] == identity
        assert actor != identity
        assert actor in item['text']
        assert identity in item['text']
    archive = next(row for row in rows if 'archive_sound_rights' in row['id'])
    arc201 = next(record['text'] for record in archive['_audit']['records'] if record['id'] == 'ARC-201')
    assert 'Speaker release SR-ARC-201 signed by Mara Velin, the recorded speaker' in arc201
    community = next(row for row in rows if 'community_energy_match' in row['id'])
    com208 = next(record['text'] for record in community['_audit']['records'] if record['id'] == 'COM-208')
    assert 'building association Hill Lantern Co-op' in com208
    assert 'Roof engineer Nia Bell' in com208


def test_v15_role_actor_schema_and_output_revision_are_explicit():
    assert v15.REV == 'authored-semantic-source-worlds-v15/11'
    assert v15.SPECS['revision'] == v15.REV
    assert v15.OUT.name.endswith('candidate-v11')
    rows = v15.make_nested_rows()
    roles = [record for row in rows for record in row['_audit']['records'] if record['role']]
    assert roles
    assert all(record['role_actor'] and record['role_actor'] != record['identity'] for record in roles)


def test_v15_candidate_output_binds_current_builder_specs_and_source(tmp_path):
    output = tmp_path / 'candidate'
    v15.main(output)
    manifest = json.loads((output / 'source-manifest.json').read_text())
    proof = json.loads((output / 'scripted-source-proof.json').read_text())
    source = (output / 'source.cases.jsonl').read_bytes()
    assert manifest['campaign'] == 'successor-v15-prepared-20261007-candidate-v11'
    assert manifest['source_revision'] == v15.REV
    assert manifest['builder_sha256'] == v15.hashlib.sha256(v15.Path(v15.__file__).read_bytes()).hexdigest()
    assert manifest['specs_sha256'] == v15.hashlib.sha256(v15.SPECS_PATH.read_bytes()).hexdigest()
    assert manifest['source_cases_sha256'] == proof['source_cases_sha256']
    assert manifest['source_cases_sha256'] == v15.hashlib.sha256(source).hexdigest()
    assert proof['model_calls'] == proof['provider_calls'] == proof['teacher_trajectories'] == 0
    assert proof['admission_granted'] is False


def test_v14_candidate_has_independent_worlds_and_sha_bound_source_proof(tmp_path):
    rows = builder.build_rows()
    stats = builder.validate_world_rows([{key: value for key, value in row.items() if key != "_audit"}
                                         for row in rows])
    assert stats == {"world_count": 6, "train_count": 3, "test_count": 3, "unique_groups": 6}
    ids = [row["id"] for row in rows]
    groups = [row["source_groups"][0] for row in rows]
    assert len(set(ids)) == len(ids)
    assert len(set(groups)) == len(groups)
    assert all(group.startswith("v14:") for group in groups)
    assert {row["split"] for row in rows} == {"train", "test"}
    builder.main(tmp_path)
    candidate = tmp_path
    source = (candidate / "source.cases.jsonl").read_bytes()
    manifest = json.loads((candidate / "source-manifest.json").read_text())
    proof = json.loads((candidate / "scripted-source-proof.json").read_text())
    assert manifest["source_cases_sha256"] == proof["source_cases_sha256"]
    assert manifest["source_cases_sha256"] == builder.hashlib.sha256(source).hexdigest()
    assert proof["admission_granted"] is False
    assert proof["model_calls"] == proof["provider_calls"] == proof["teacher_trajectories"] == 0


def test_nested_reducers_bind_inherited_local_and_item_scoped_evidence():
    rows = builder.build_rows()
    reducers = [row for row in rows if row["_audit"]["slice_kind"] == "nested-reducer"]
    assert len(reducers) == 2
    for row in reducers:
        sem = row["semantics"]
        code = row["curriculum"]["reference"]["root"][0][1]["code"]
        assert "folder.folders('teams/*')" in code
        assert "teams.map(reduceTeam)" in code
        assert "inheritedRule" in code and "localRule" in code
        assert "item:FileHandle" in code
        assert "do not borrow evidence from a sibling folder or item" in code
        audit = row["_audit"]["records"]
        assert len(audit) == 6
        assert sum(item["accepted"] for item in audit) == 3
        for item in audit:
            assert len(item["clauses"]) == 3
            if item["accepted"]:
                assert all(item["clauses"]) and item["failed_clause"] is None
            else:
                assert item["clauses"].count(False) == 1
                assert item["failed_clause"] == item["clauses"].index(False)
        policies = [path for path in sem["folder_files"] if path.endswith("/policy.json")]
        overrides = [path for path in sem["folder_files"] if path.endswith("/override.json")]
        items = [path for path in sem["folder_files"] if "/items/" in path]
        assert len(policies) == len(overrides) == 2 and len(items) == 6
        assert all(path in sem["folder_files"] for path in policies + overrides + items)
    field_station = next(row for row in reducers if "field_station_access" in row["id"])
    ordered = sorted(field_station["_audit"]["records"], key=lambda item: item["id"])
    assert [item["accepted"] for item in ordered] == [False, True, False, True, True, False]


def test_iterate_worlds_carry_full_draft_for_four_or_five_evidence_passes():
    rows = builder.build_rows()
    iterates = [row for row in rows if row["_audit"]["slice_kind"] == "iterate"]
    assert len(iterates) == 4
    pass_counts = set()
    for row in iterates:
        audit = row["_audit"]
        passes = audit["passes"]
        pass_counts.add(len(passes))
        assert len(passes) in (4, 5)
        assert len(audit["field_names"]) == len(passes)
        previous = dict(audit["initial"])
        for step in passes:
            current = step["state"]
            changed = [key for key in current if current[key] != previous[key]]
            assert changed == [step["field"]]
            packet = row["semantics"]["folder_files"]["packet.md"].casefold()
            assert current[step["field"]].casefold() in packet
            assert step["name"] in row["semantics"]["folder_files"]["task.json"]
            previous = current
        assert previous == audit["final"] == row["semantics"]["expected"]
        code = row["curriculum"]["reference"]["root"][0][1]["code"]
        assert "iterateOn(revise" in code and "complete current draft" in code
        assert "maxSteps:task.passes.length" in code
        children = row["curriculum"]["reference"]["children"]
        assert len(children) == len(passes)
        assert all(len(child["calls"]) == 2 for child in children)
        assert all(passes[index]["name"] in child["match"] for index, child in enumerate(children))
    assert pass_counts == {4, 5}


def test_runtime_references_match_scoped_files_and_return_exact_expected_values():
    rows = builder.build_rows()
    for row in rows:
        reference = row["curriculum"]["reference"]
        expected = row["semantics"]["expected"]
        root_returns = [args["value"] for tool, args in reference["root"] if tool == "return_result"]
        assert root_returns == [expected]
        if row["_audit"]["slice_kind"] == "nested-reducer":
            files = row["semantics"]["folder_files"]
            records = row["_audit"]["records"]
            children = reference["children"]
            assert len(children) == len(records) == 6
            for record, child in zip(records, children):
                leaf = f"{record['id']}.md"
                source_paths = [path for path in files if path.endswith(f"/items/{leaf}")]
                assert len(source_paths) == 1
                assert child["match"] == leaf
                assert child["calls"][0] == ["read_file", {"path": leaf}]
                assert child["calls"][1][1]["value"] is record["accepted"]
        else:
            passes = row["_audit"]["passes"]
            children = reference["children"]
            assert len(children) == len(passes)
            for step, child in zip(passes, children):
                assert step["name"] in child["match"]
                assert child["calls"][0] == ["read_file", {"path": "packet.md"}]
                assert child["calls"][1][1]["value"] == step["state"]
