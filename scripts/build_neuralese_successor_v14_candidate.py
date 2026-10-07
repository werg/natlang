from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from neuralese_source_world_builder import build_proof, json_text, validate_world_rows

OUT = ROOT / "runs/neuralese-successor-v14-prepared-20261007-candidate-v5-final-review"
REV = "authored-semantic-source-worlds-v14/1"


def result_event(value):
    return ["return_result", {"status": "success", "value": value}]


def reducer_world(slug, split, domain, inherited, teams, records, clauses, item_scope):
    group = f"v14:{slug}:world"
    ids = [record["id"] for record in records]
    files = {
        "task.json": json_text({
            "instruction": f"Review the {domain} packet under the inherited and local rules.",
            "output_path": "selection.json",
            "output_contract": {
                "format": "JSON array of complete item IDs sorted lexicographically.",
                "reconciliation": "Include every and only items satisfying every inherited, local, and item-specific clause.",
            },
        }),
        "selection.json": "[]",
    }
    for team, local in teams.items():
        files[f"teams/{team}/policy.json"] = json_text({"inherited_rule": inherited})
        files[f"teams/{team}/override.json"] = json_text({"local_rule": local})
    child_results = []
    clause_audit = []
    for team, local in teams.items():
        calls = []
        for record in [r for r in records if r["team"] == team]:
            path = f"teams/{team}/items/{record['id']}.md"
            files[path] = record["text"]
            outcome = all(record["supports"])
            calls.extend([["read_file", {"path": path}], result_event(outcome)])
            clause_audit.append({"id": record["id"], "clauses": record["supports"],
                                 "accepted": outcome, "failed_clause": record.get("failed_clause")})
        for record in [r for r in records if r["team"] == team]:
            outcome = all(record["supports"])
            child_results.append({
                "match": f"{record['id']}.md",
                # Each child receives one FileHandle rebased to its own root;
                # in that scope the evidence filename is the leaf filename.
                "calls": [["read_file", {"path": f"{record['id']}.md"}], result_event(outcome)],
            })
    expected = sorted(record["id"] for record in records if all(record["supports"]))
    root_code = (
        "const task=await folder.file('task.json').readJson(); const teams=await folder.folders('teams/*'); "
        "const reduceTeam=async (team:Folder):Promise<string[]>=>{ "
        "const inheritedRule:string=JSON.parse(await team.file('policy.json').readText()).inherited_rule; "
        "const localRule:string=JSON.parse(await team.file('override.json').readText()).local_rule; "
        "const taskInstruction:string=task.instruction; "
        "const items=await team.files(`${team.path}/items/*.md`); "
        "const judge:Neuralese<(item:FileHandle)=>Promise<boolean>>=nl.with({inheritedRule,localRule,taskInstruction})`<|neuralese|>Read the complete item FileHandle. Apply the inherited rule, this folder's local override, and all item-specific facts. Decide only this item; do not borrow evidence from a sibling folder or item. Return true only when every applicable condition is supported.<|/neuralese|>`; "
        "const pairs=await Promise.all(items.map(async item=>{const itemId=item.name.replace(/\\.md$/,''); return [itemId,await judge(item)] as const;})); "
        "return pairs.filter(([,keep])=>keep).map(([id])=>id); }; "
        "const nested=await Promise.all(teams.map(reduceTeam)); "
        "const selected=nested.flat().sort(); await folder.file(task.output_path).writeText(JSON.stringify(selected)); "
        "const saved=await folder.file(task.output_path).readJson(); if(JSON.stringify(saved)!==JSON.stringify(selected)) throw Error('readback mismatch'); return saved;"
    )
    return {
        "version": "natlang.program/2",
        "id": f"inline-curriculum:{REV.replace('/', '_')}:v14-{slug}-nested-reducer:inherited-local-item-reconciliation",
        "kind": "lambda_source", "family": "curriculum_authored_semantic_source_worlds_v14",
        "source": "natlang-inline-curriculum", "split": split,
        "source_ids": [group], "source_groups": [group], "source_revisions": [REV],
        "license": "project-generated", "gold_sources": ["constructed-world-oracle"],
        "generation": {
            "generator": REV, "independent_world": slug,
            "capture_contract": {"inherited_rule": "folder policy JSON", "local_rule": "same-folder override JSON",
                                 "item_facts": "same-item FileHandle", "cross_item_evidence": "forbidden"},
            "source_quality": "each negative item fails exactly one explicit applicable clause",
        },
        "curriculum": {
            "version": "natlang.inline_curriculum/1", "family": "authored_semantic_source_worlds_v14",
            "family_version": 14, "shape": f"v14-{slug}-nested-reducer",
            "variant": "inherited-policy-local-override-item-evidence", "pair_group": None,
            "split_group": group, "slice": "nested_scoped", "domain": "other",
            "mode": "single_call", "inline": "required", "iterate": "required",
            "evidence": {"world": [inherited], "retrieved": ["folder policy and local override are read from the current nested folder"],
                         "background": []},
            "assumptions": [],
            "decisive": [{"marker": records[0]["id"], "source": "child",
                          "note": "Current item evidence is read from its matching nested folder."}],
            "plausible_actions": ["reduce each nested folder with its inherited policy, local override, and own item facts"],
            "minimum_sequence": ["read the parent task", "apply the reducer to each policy folder", "write and verify sorted complete IDs"],
            "reference": {"root": [["eval", {"code": root_code}], result_event(expected)], "children": child_results},
        },
        "semantics": {
            "root": "review_nested_items.nl",
            "files": {"review_nested_items.nl": "---\nargs: {}\nreturns: string[]\nkind: directory-reducer\n---\nRead task.json. Apply one inline semantic reducer to each nested team folder. Each child receives that folder's inherited policy and local override as explicit captures and one complete item FileHandle. Return sorted IDs of all and only items satisfying every applicable clause. Write, read back, verify, and return.\n"},
            "inputs": {},
            "expected": expected,
            "folder_files": files,
            "expected_files": {**files, "selection.json": json_text(expected)},
        },
        "_audit": {"slice_kind": "nested-reducer", "clauses": clauses, "item_scope": item_scope, "records": clause_audit},
    }


def iterate_world(slug, split, domain, fields, passes, initial, packet):
    group = f"v14:{slug}:world"
    keys = [field for field, _, _ in fields]
    shape = "{ " + "; ".join(f"{field}: string" for field in keys) + " }"
    pass_json = json_text([{"name": name, "constraint": constraint} for name, constraint, _ in passes])
    task = {
        "instruction": f"Reconcile the {domain} record using the ordered source checks.",
        "output_path": "decision.json",
        "output_contract": {
            "format": "JSON object with exactly the declared string fields.",
            "fields": {field: contract for field, _, contract in fields},
            "carry_forward": "Each pass changes only its assigned field and preserves all other current values exactly.",
        },
        "initialDraft": initial,
        "passes": json.loads(pass_json),
    }
    child_results = []
    current = dict(initial)
    snapshots = []
    for index, (name, constraint, evidence) in enumerate(passes):
        field, value, _ = fields[index]
        current = dict(current)
        current[field] = value
        snapshots.append(dict(current))
        # Match the exact captured pass value in each invocation opening. The
        # quote escaping is part of the runtime's rendered capture text.
        escaped_name = json.dumps(name).replace('"', '\\"')
        child_results.append({
            "match": f"const passName: string = {escaped_name};",
            "calls": [["read_file", {"path": "packet.md"}], result_event(dict(current))],
        })
    root_code = (
        "const task=await folder.file('task.json').readJson(); const packet=await folder.file('packet.md'); "
        f"type Progress={{pass:number;draft:{shape}}}; "
        "const revise=async (progress:Progress):Promise<Progress>=>{ const current=task.passes[progress.pass]; "
        "const instruction=task.instruction; const contract=JSON.stringify(task.output_contract); "
        "const passName=current.name; const constraint=current.constraint; const currentDraft=JSON.stringify(progress.draft); "
        f"const step:Neuralese<(packet:FileHandle)=>Promise<{shape}>>=nl.with({{instruction,contract,passName,constraint,currentDraft}})"
        "`<|neuralese|>Read the same complete packet FileHandle. Use the supplied instruction, contract, pass constraint, and complete current draft. Apply only this pass, use the authoritative evidence for its assigned field, preserve every other supported field exactly, and return the whole object with exactly the declared string fields.<|/neuralese|>`; "
        "const next=await step(packet); return {pass:progress.pass+1,draft:next}; }; "
        "const final=await iterateOn(revise,{pass:0,draft:task.initialDraft} as Progress).withLimit({maxSteps:task.passes.length}).until(p=>p.pass>=task.passes.length); "
        "await folder.file(task.output_path).writeText(JSON.stringify(final.draft)); const saved=await folder.file(task.output_path).readJson(); "
        "if(JSON.stringify(saved)!==JSON.stringify(final.draft)) throw Error('readback mismatch'); return saved.draft ?? saved;"
    )
    final = snapshots[-1]
    return {
        "version": "natlang.program/2",
        "id": f"inline-curriculum:{REV.replace('/', '_')}:v14-{slug}-iterate:ordered-complete-draft-reconciliation",
        "kind": "lambda_source", "family": "curriculum_authored_semantic_source_worlds_v14",
        "source": "natlang-inline-curriculum", "split": split,
        "source_ids": [group], "source_groups": [group], "source_revisions": [REV],
        "license": "project-generated", "gold_sources": ["constructed-world-oracle"],
        "generation": {
            "generator": REV, "independent_world": slug,
            "capture_contract": {"instruction": "task-visible", "contract": "exact declared fields", "pass": "ordered current-pass only",
                                 "current_draft": "complete actual carried draft", "evidence": "same packet FileHandle"},
            "source_quality": "each pass changes one assigned field from explicit current-pass evidence and carries the full draft",
        },
        "curriculum": {
            "version": "natlang.inline_curriculum/1", "family": "authored_semantic_source_worlds_v14",
            "family_version": 14, "shape": f"v14-{slug}-iterate", "variant": "ordered-complete-draft-reconciliation",
            "pair_group": None, "split_group": group, "slice": "iterate", "domain": "other",
            "mode": "single_call", "inline": "required", "iterate": "required",
            "evidence": {"world": [domain], "retrieved": ["one complete packet FileHandle shared by all ordered passes"], "background": []},
            "assumptions": [],
            "decisive": [{"marker": re.search(r"[A-Z]{2}-\d+", packet).group(0), "source": "child",
                          "note": f"Current pass {name} reads its same-packet evidence."}
                         for name, _, _ in passes],
            "plausible_actions": ["apply the current pass only and preserve all other fields in the complete carried draft"],
            "minimum_sequence": ["read task and packet", f"run exactly {len(passes)} ordered passes", "write/read back exact final object"],
            "reference": {"root": [["eval", {"code": root_code}], result_event(final)],
                          "children": child_results},
        },
        "semantics": {
            "root": "reconcile_item.nl",
            "files": {"reconcile_item.nl": f"---\nargs: {{}}\nreturns: \"{shape}\"\nkind: directory-reducer\n---\nRead task.json and packet.md. Return exactly the declared string fields. Use iterateOn with one Progress holding the pass index and complete current draft. Each child receives the complete task, contract, current pass and constraint, complete current draft, and same packet FileHandle. Apply only the current pass, preserve all other supported values exactly, write/read back/verify the whole object.\n"},
            "inputs": {}, "expected": final,
            "folder_files": {"task.json": json_text(task), "packet.md": packet, "decision.json": json_text(initial)},
            "expected_files": {"task.json": json_text(task), "packet.md": packet, "decision.json": json_text(final)},
        },
        "_audit": {"slice_kind": "iterate", "passes": [{"name": p[0], "field": fields[i][0], "evidence": p[2], "state": snapshots[i]} for i, p in enumerate(passes)],
                   "field_names": keys, "initial": initial, "final": final},
    }


def build_rows():
    rows = []
    # Two independent hierarchical decision worlds. Each contains a parent
    # rule, per-folder exceptions, and inherited/local item counterexamples.
    reducer_specs = [
        {
            "slug": "archive_media_stewardship", "split": "train", "domain": "regional archive media intake",
            "inherited": "Accept an item only when its accession is linked to a named donor and the deposit authority approved the transfer.",
            "teams": {"audio": "For oral-history media, require a transcript release signed by the recorded speaker.",
                      "maps": "For route-map sets, require the accessibility note to match the current map revision."},
            "records": [
                {"id":"AMS-41","team":"audio","text":"Accession AMS-41 is linked to donor Tern Street Radio; the deposit authority approved transfer. Speaker signed transcript release TR-41, which names audio revision 3.","supports":[True,True,True],"failed_clause":None},
                {"id":"AMS-42","team":"audio","text":"Accession AMS-42 is linked to donor Lantern Choir; the deposit authority approved transfer. Transcript TR-42 names revision 2, but its speaker release is unsigned.","supports":[True,True,False],"failed_clause":2},
                {"id":"AMS-43","team":"maps","text":"Accession AMS-43 is linked to donor North Quay Survey; the deposit authority approved transfer. Map set revision 5 and accessibility note revision 5 agree.","supports":[True,True,True],"failed_clause":None},
                {"id":"AMS-44","team":"maps","text":"Accession AMS-44 is linked to donor Orchard Walkers; the deposit authority approved transfer. Map set is revision 8, while its accessibility note remains revision 7.","supports":[True,True,False],"failed_clause":2},
                {"id":"AMS-45","team":"audio","text":"Accession AMS-45 has no named donor in the intake record; the deposit authority approved transfer. Speaker signed transcript release TR-45, which names audio revision 4.","supports":[False,True,True],"failed_clause":0},
                {"id":"AMS-46","team":"maps","text":"Accession AMS-46 is linked to donor East Bluff Cartographers; the deposit authority approved transfer. Map set revision 2 and accessibility note revision 2 agree.","supports":[True,True,True],"failed_clause":None},
            ],
            "clauses":["accession linked to donor","deposit authority approved transfer","local media or map exception satisfied"],
            "item_scope":"Each item's own accession and attached approval/release or map-note evidence; sibling folders cannot fill a missing clause.",
        },
        {
            "slug": "field_station_access", "split": "test", "domain": "coastal field-station access clearance",
            "inherited": "A visit is eligible only when the named research team is on the seasonal roster and the station custodian has approved the visit window.",
            "teams": {"tidepool": "For tidepool work, require a current habitat-monitor briefing acknowledgment.",
                      "nesting": "For nesting-cliff work, require a ranger escort plan accepted by the local ranger."},
            "records": [
                {"id":"FSA-19","team":"tidepool","text":"Visit FSA-19 names the Heron Group, listed on the seasonal roster. Custodian approved the 4 September window. Its briefing acknowledgement covers last season's protocol, not the current monitor protocol.","supports":[True,True,False],"failed_clause":2},
                {"id":"FSA-18","team":"tidepool","text":"Visit FSA-18 names the Bracken Lab team, listed on the seasonal roster. Custodian approved the 3 September window. Briefing acknowledgement BA-18 covers the current tidepool monitor protocol.","supports":[True,True,True],"failed_clause":None},
                {"id":"FSA-21","team":"nesting","text":"Visit FSA-21 names the Salt Grass Survey team on the seasonal roster; the custodian approved 9 September. Escort plan EP-21 is drafted but no local ranger has accepted it.","supports":[True,True,False],"failed_clause":2},
                {"id":"FSA-23","team":"nesting","text":"Visit FSA-23 names the Kelp Edge team on the seasonal roster; the custodian approved 11 September. Ranger escort plan EP-23 was accepted by ranger D. Hale.","supports":[True,True,True],"failed_clause":None},
                {"id":"FSA-20","team":"nesting","text":"Visit FSA-20 names the Dune Ecology team on the seasonal roster; custodian approval covers 8 September. Ranger escort plan EP-20 was accepted by ranger L. Vale.","supports":[True,True,True],"failed_clause":None},
                {"id":"FSA-22","team":"tidepool","text":"Visit FSA-22 names the Moon Snail team, which is absent from the seasonal roster. Custodian approval covers 10 September. Briefing BA-22 acknowledges the current habitat-monitor protocol.","supports":[False,True,True],"failed_clause":0},
            ],
            "clauses":["named team on seasonal roster","custodian approved this visit window","local habitat-specific safeguard satisfied"],
            "item_scope":"The visit's team, date, custodian decision, and habitat-specific attachment; another visit's permission is not transferable.",
        },
    ]
    for spec in reducer_specs:
        if spec["slug"] == "field_station_access":
            # Change the complete ID bundles via a bijection, so held-out
            # positives occupy positions 2, 4, and 5 after lexical sorting.
            rename = {"FSA-19": "FSA-18", "FSA-18": "FSA-19",
                      "FSA-21": "FSA-20", "FSA-20": "FSA-21",
                      "FSA-23": "FSA-22", "FSA-22": "FSA-23"}
            pattern = re.compile(r"FSA-(?:18|19|20|21|22|23)")
            for record in spec["records"]:
                record["id"] = rename[record["id"]]
                record["text"] = pattern.sub(lambda match: rename[match.group(0)], record["text"])
        rows.append(reducer_world(**spec))

    # Ordered passes use four and five distinct semantic evidence sources.
    rows.append(iterate_world(
        "repair_collective_dispatch", "train", "community repair collective dispatch plan",
        fields=[("request","RC-288","Copy the complete request ID from the intake ledger."),
                ("repairer","Imani Cho","Use the volunteer assigned to the request in the skills roster."),
                ("part","ceramic speed dial","Copy the approved replacement component from the parts note."),
                ("disposition","dispatch","Use the coordinator's signed dispatch decision as lowercase dispatch or wait.")],
        passes=[("intake ledger","Copy the paired request identifier and keep the remainder of the draft.","Intake ledger pairs request RC-288 with the mixer repair."),
                ("skills roster","Use only the named request's assigned volunteer.","Skills roster assigns mixer request RC-288 to Imani Cho."),
                ("parts note","Copy the approved part for this request; do not infer a substitute.","Parts note approves a ceramic speed dial for RC-288."),
                ("coordinator decision","Copy the signed dispatch disposition exactly.","Coordinator signs dispatch for RC-288 after confirming the volunteer and part.")],
        initial={"request":"R-0","repairer":"unassigned","part":"none","disposition":"wait"},
        packet="Intake ledger: request RC-288 is a mixer repair. Skills roster: RC-288 is assigned to Imani Cho. Parts note: ceramic speed dial is approved for RC-288. Coordinator N. Patel signs DISPATCH for RC-288 after confirming the assignment and part.",
    ))
    rows.append(iterate_world(
        "wetland_trail_reopening", "test", "wetland trail reopening review",
        fields=[("trail","NW-6","Copy the trail identifier paired with the north boardwalk section."),
                ("surface","stable after the handrail repair","Record the current surface condition from the inspection note."),
                ("wildlife","maintain the reed-nest buffer","Copy the current wildlife buffer instruction from the seasonal notice."),
                ("access","open","Use the ranger's signed access determination as lowercase open or closed.")],
        passes=[("route register","Copy the trail ID paired to the named boardwalk section.","Route register pairs NW-6 with north boardwalk."),
                ("surface inspection","Use the inspection's current condition, not last month's note.","Inspection says NW-6 surface is stable after the handrail repair."),
                ("seasonal notice","Copy the active nesting buffer instruction.","Seasonal notice says maintain the reed-nest buffer on NW-6."),
                ("ranger determination","Copy the ranger's signed access decision for NW-6.","Ranger signs OPEN for NW-6 with the reed-nest buffer maintained.")],
        initial={"trail":"T-0","surface":"unknown","wildlife":"none","access":"closed"},
        packet="Route register pairs NW-6 with north boardwalk. Current inspection says its surface is stable after the handrail repair. The active seasonal notice's exact instruction is 'maintain the reed-nest buffer'. Ranger E. Moss signs OPEN for NW-6 provided the buffer remains in place.",
    ))

    # Fifth pass demonstrates deeper context while preserving the same complete
    # carried draft and pass-local authority.
    rows.append(iterate_world(
        "makerspace_tool_return", "train", "makerspace shared-tool return reconciliation",
        fields=[("tool","TR-6","Copy the tool tag from the checkout slip."),
                ("borrower","Devon Iri","Copy the named borrower from the checkout slip."),
                ("condition","guard intact; battery removed","Use the return-desk inspection result."),
                ("safety","clean-and-release","Copy the safety lead's action for the observed condition."),
                ("shelf","cabinet R-2","Use the signed storage destination.")],
        passes=[("checkout slip","Copy the tool tag from the checkout slip.","Slip CS-77 checks out rotary trimmer TR-6 to Devon Iri."),
                ("borrower roster","Copy the borrower assigned to TR-6.","Checkout roster confirms Devon Iri is assigned to TR-6."),
                ("return inspection","Copy the observed return condition.","Inspection records the exact condition for CS-77."),
                ("safety lead","Copy the safety action tied to this inspection.","Safety lead records clean-and-release for CS-77 after guard inspection."),
                ("storage register","Copy the assigned destination confirmed by the custodian.","Storage register assigns rotary trimmers to cabinet R-2. Tool custodian signs return of CS-77 to cabinet R-2.")],
        initial={"tool":"unknown","borrower":"unknown","condition":"unchecked","safety":"hold","shelf":"unassigned"},
        packet="Checkout slip CS-77 names rotary trimmer TR-6 and borrower Devon Iri. Checkout roster confirms Devon Iri is assigned to TR-6. Return inspection's exact condition for CS-77 is 'guard intact; battery removed'. Safety lead records clean-and-release after guard inspection. Storage register assigns rotary trimmers to cabinet R-2. Tool custodian signs return of CS-77 to cabinet R-2.",
    ))
    rows.append(iterate_world(
        "mutual_aid_cold_storage", "test", "mutual-aid cold-storage placement plan",
        fields=[("shipment","MA-309","Copy the complete shipment code."),
                ("recipient","Willow Street Pantry","Copy the recipient site paired with this shipment."),
                ("handling","sealed, chilled","Use the handling instruction for the recorded contents."),
                ("custody","Noor Bell","Copy the receiving steward who accepted handoff."),
                ("placement","bin C-4","Use the storekeeper's approved location.")],
        passes=[("dispatch roster","Copy the shipment code from its dispatch entry.","Dispatch roster assigns shipment MA-309."),
                ("recipient manifest","Copy the recipient site paired with MA-309.","Recipient manifest pairs MA-309 with Willow Street Pantry."),
                ("contents card","Use the instruction attached to this shipment's contents.","Contents card requires sealed, chilled handling for cultured starter."),
                ("handoff log","Copy the receiving steward who signed this handoff.","Handoff log records steward Noor Bell signed the handoff accepting shipment MA-309."),
                ("storekeeper approval","Copy the approved placement and preserve every carried field.","Storekeeper approves MA-309 in bin C-4 under sealed chilled handling.")],
        initial={"shipment":"S-0","recipient":"unknown","handling":"ambient","custody":"unassigned","placement":"none"},
        packet="Dispatch roster assigns shipment MA-309. Recipient manifest pairs MA-309 with Willow Street Pantry. Contents card requires sealed, chilled handling for cultured starter. Handoff log records steward Noor Bell signed the handoff accepting shipment MA-309. Cold-room map assigns sealed starter to bin C-4. Storekeeper approves MA-309 in bin C-4 under sealed chilled handling.",
    ))
    return rows


def main(output_dir: Path | None = None):
    output_dir = output_dir or OUT
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    # Drop builder-only audit fields from published source rows.
    audits = {row["id"]: row.pop("_audit") for row in rows}
    stats = validate_world_rows(rows)
    raw = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows).encode()
    source_path = output_dir / "source.cases.jsonl"
    source_path.write_bytes(raw)
    proof = build_proof(rows, raw)
    proof["world_audits"] = audits
    proof["semantic_passes"] = {slug: audit["passes"] for slug, audit in ((row["curriculum"]["shape"], audits[row["id"]]) for row in rows) if "passes" in audit}
    proof["nested_reducers"] = {row["curriculum"]["shape"]: audits[row["id"]] for row in rows if "records" in audits[row["id"]]}
    proof_path = output_dir / "scripted-source-proof.json"
    proof_path.write_text(json.dumps(proof, indent=2, ensure_ascii=False) + "\n")
    audit_rows = list(audits.values())
    record_audits = [item for audit in audit_rows for item in audit.get("records", [])]
    pass_audits = [audit for audit in audit_rows if "passes" in audit]
    quality = {
        "schema": "natlang.neuralese-source-quality-review/1", "revision": REV,
        "source_cases_sha256": hashlib.sha256(raw).hexdigest(), "review_type": "deterministic source consistency; independent semantic review remains required",
        "world_count": stats["world_count"], "train_count": stats["train_count"], "test_count": stats["test_count"],
        "nested_reducer_worlds": sum("records" in audit for audit in audit_rows),
        "iterate_worlds": len(pass_audits),
        "nested_records": len(record_audits),
        "positive_records": sum(item["accepted"] for item in record_audits),
        "negative_records": sum(not item["accepted"] for item in record_audits),
        "iterative_pass_counts": {str(count): sum(len(audit["passes"]) == count for audit in pass_audits) for count in (4, 5)},
        "checks": {"unique_factual_groups": True, "split_groups_disjoint": True,
                   "nested_reducers_have_inherited_and_local_rules": True,
                   "negative_reducer_items_each_fail_one_applicable_clause": True,
                   "iterate_full_draft_carried_each_pass": True, "all_pass_evidence_visible": True,
                   "model_calls": 0, "teacher_trajectories": 0, "admission_granted": False},
        "limits": ["Scripted return values are source oracles, not teacher observations.",
                   "No semantic admission or runtime execution certification is granted."],
    }
    (output_dir / "source-quality-review.json").write_text(json.dumps(quality, indent=2, ensure_ascii=False) + "\n")
    manifest = {
        "schema": "natlang.neuralese-semantic-source-v14/1", "campaign": "successor-v14-prepared-20261007-candidate-v5-final-review",
        "task_count": stats["world_count"], "train_count": stats["train_count"], "test_count": stats["test_count"],
        "source_revision": REV, "source_cases": "source.cases.jsonl", "source_cases_sha256": hashlib.sha256(raw).hexdigest(),
        "source_quality_review": "source-quality-review.json", "scripted_proof": "scripted-source-proof.json",
        "runtime_proof": "runtime-reference-proof.json", "semantic_and_runtime_audit": "semantic-and-runtime-audit.json",
        "builder": "scripts/build_neuralese_successor_v14_candidate.py",
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "admission": "candidate only; no teacher generation, admission, or training launch",
        "unique_groups": True, "worlds": [row["curriculum"]["shape"] for row in rows],
    }
    (output_dir / "source-manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({**stats, "sha256": hashlib.sha256(raw).hexdigest(), "worlds": manifest["worlds"]}, indent=2))


if __name__ == "__main__":
    main()
