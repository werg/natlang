import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "scripts/build_neuralese_successor_v15_candidate.py"
SPEC = importlib.util.spec_from_file_location("v15_source_builder", BUILDER)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_v15_has_24_independent_worlds_with_balanced_splits_and_unique_groups():
    rows = builder.build_rows()
    stats = builder.validate_world_rows([{key: value for key, value in row.items() if key != "_audit"}
                                         for row in rows])
    assert stats == {"world_count": 24, "train_count": 12, "test_count": 12, "unique_groups": 24}
    assert len({row["id"] for row in rows}) == 24
    assert all(row["source_revisions"] == [builder.REV] for row in rows)
    assert sum("nested-reducer" in row["curriculum"]["shape"] for row in rows) == 12
    assert sum("-iterate" in row["curriculum"]["shape"] for row in rows) == 12


def test_nested_worlds_use_explicit_world_specific_facts_and_one_clause_negatives():
    rows = builder.make_nested_rows()
    assert len(rows) == 12
    rank_counts = {split: [0] * 6 for split in ("train", "test")}
    for row in rows:
        records = row["_audit"]["records"]
        assert len(records) == 6
        assert sum(record["accepted"] for record in records) == 3
        assert len(row["source_groups"]) == 1
        assert row["source_groups"][0] not in {
            prior["source_groups"][0] for prior in rows if prior is not row
        }
        files = row["semantics"]["folder_files"]
        item_text = "\n".join(text for path, text in files.items() if "/items/" in path)
        assert any(term in item_text.casefold() for term in ("does not", "no ", "did not"))
        assert any(term in item_text for term in ("deposit release", "approved", "signed repair authorization", "public-use release"))
        assert all(path in files for path in files if path.endswith("/policy.json") or path.endswith("/override.json"))
        ordered = sorted(records, key=lambda record: record["id"])
        for rank, record in enumerate(ordered):
            rank_counts[row["split"]][rank] += int(record["accepted"])
            assert len(record["clauses"]) == 3
            if record["accepted"]:
                assert all(record["clauses"]) and record["failed_clause"] is None
            else:
                assert record["clauses"].count(False) == 1
                assert record["failed_clause"] == record["clauses"].index(False)
        assert sorted(row["semantics"]["expected"]) == sorted(
            record["id"] for record in records if record["accepted"])
        children = row["curriculum"]["reference"]["children"]
        assert len(children) == 6
        for child in children:
            assert child["match"].endswith(".md")
            assert child["calls"][0] == ["read_file", {"path": child["match"]}]
    assert all(1 <= count <= 4 for counts in rank_counts.values() for count in counts)
    assert all(rank_counts["train"][rank] != 0 and rank_counts["test"][rank] != 0 for rank in range(6))


def test_iterate_worlds_change_one_gold_grounded_field_per_pass_and_keep_full_draft():
    rows = builder.make_iterate_rows()
    assert len(rows) == 12
    pass_distribution = {4: 0, 5: 0, 6: 0}
    for row in rows:
        audit = row["_audit"]
        passes = audit["passes"]
        assert len(passes) in pass_distribution
        pass_distribution[len(passes)] += 1
        assert len(audit["field_names"]) == len(passes)
        packet = row["semantics"]["folder_files"]["packet.md"].casefold()
        previous = dict(audit["initial"])
        for index, step in enumerate(passes):
            state = step["state"]
            assert set(state) == set(audit["field_names"])
            assert [key for key in state if state[key] != previous[key]] == [step["field"]]
            assert state[step["field"]].casefold() in packet
            assert step["evidence"].casefold() in packet
            child = row["curriculum"]["reference"]["children"][index]
            assert step["name"] in child["match"]
            assert child["calls"][0] == ["read_file", {"path": "packet.md"}]
            assert child["calls"][1][1]["value"] == state
            previous = state
        assert previous == audit["final"] == row["semantics"]["expected"]
        code = row["curriculum"]["reference"]["root"][0][1]["code"]
        assert "iterateOn(revise" in code
        assert "const currentDraft=JSON.stringify(progress.draft)" in code
        assert "return the whole object" in code
        assert row["split"] in ("train", "test")
    assert pass_distribution == {4: 4, 5: 4, 6: 4}


def test_categorical_iterate_fields_require_exact_tokens_without_explanations():
    rows = builder.make_iterate_rows()
    required = {
        "status": ("approved or denied", "omit explanation"),
        "safety": ("clean-and-release or hold", "omit timing"),
        "rights": ("public release or hold", "omit explanation"),
        "access": ("open with buffer or closed", "omit explanation"),
        "collection": ("no collection or collection permitted", "omit explanation"),
    }
    found = {}
    for row in rows:
        contract = row["semantics"]["folder_files"]["task.json"]
        import json
        fields = json.loads(contract)["output_contract"]["fields"]
        for key, value in fields.items():
            if key in required:
                found[key] = value.casefold()
    assert found.keys() == required.keys()
    for key, (choices, no_reason) in required.items():
        assert "only the" in found[key] and "token" in found[key]
        assert choices in found[key]
        assert no_reason in found[key]


def test_nested_item_contains_its_local_rule_applicability_category():
    def expected_category(world, item):
        thing, site = item["thing"].casefold(), item["site"].casefold()
        if world == "archive_sound_rights":
            return "oral_history" if "interview reel" in thing or "harvest song" in thing or "oral history" in thing else "field_recording"
        if world == "transit_accessibility_repair":
            return "vehicle" if "vehicle" in thing or "vehicle" in site or "bus" in thing else "platform"
        if world == "volunteer_cold_chain":
            return "cultures" if any(token in thing for token in ("starter jar", "tempeh culture", "sourdough mother")) else "produce"
        if world == "habitat_field_permits":
            return "cliff" if "cliff" in thing or "cliff" in site else "wetland"
        if world == "translation_editorial_clearance":
            return "Kalo" if "kalo" in site else "Ruma"
        if world == "craft_repair_material_release":
            return "textile" if any(token in thing for token in ("loom shuttle", "embroidered apron", "apron seam")) else "woodshop"
        if world == "library_oral_history_release":
            return "web_audio" if "web audio" in site else "exhibit"
        if world == "watershed_sampling_safety":
            return "upper_basin" if "upper" in site else "lower_basin"
        if world == "community_energy_match":
            return "heat_pump" if "heat-pump" in thing else "solar"
        if world == "school_accessible_trip":
            return "wheelchair_route" if "step-free" in thing else "sensory_plan"
        if world == "clinic_interpreter_roster":
            return "Kalo" if "kalo" in thing else "Ruma"
        if world == "neighborhood_tree_work":
            # Nesting status is a record qualifier ("with a confirmed active nest"), not part of the tree's name.
            return "street_tree" if thing.startswith("street") else "nesting_tree"
        raise AssertionError(world)

    for row in builder.make_nested_rows():
        world = row["curriculum"]["shape"].split("-", 1)[1].split("-nested-reducer")[0]
        files = row["semantics"]["folder_files"]
        audit = row["_audit"]["records"]
        per_team = {}
        for item in audit:
            path = next(path for path in files if path.endswith(f"/items/{item['id']}.md"))
            team = path.split("/")[1]
            text = files[path]
            assert team == expected_category(world, item)
            assert f"Classification: {team.replace('_', ' ')}." in text
            counts = per_team.setdefault(team, [0, 0])
            counts[0 if item["accepted"] else 1] += 1
        assert all(positive > 0 and negative > 0 for positive, negative in per_team.values())


def test_free_text_iterate_fields_require_exact_copy_without_paraphrase():
    import json
    rows = builder.make_iterate_rows()
    categorical = {"status", "safety", "rights", "access", "collection"}
    for row in rows:
        task = json.loads(row["semantics"]["folder_files"]["task.json"])
        fields = task["output_contract"]["fields"]
        for key, contract in fields.items():
            if key not in categorical:
                lowered = contract.casefold()
                assert "exact source wording" in lowered
                assert "without paraphrasing" in lowered
                assert "punctuation" in lowered


def test_nested_authority_fact_uses_world_specific_approval_or_signature():
    for row in builder.make_nested_rows():
        world = row["curriculum"]["shape"].split("-", 1)[1].split("-nested-reducer")[0]
        files = row["semantics"]["folder_files"]
        for item in row["_audit"]["records"]:
            path = next(path for path in files if path.endswith(f"/items/{item['id']}.md"))
            text = files[path].casefold()
            authority_sentence = item["authority_evidence"].casefold()
            assert authority_sentence in text
            assert any(verb in authority_sentence for verb in ("approved", "signed"))
            assert ("no " in authority_sentence or "did not" in authority_sentence) == (not item["clauses"][1])
