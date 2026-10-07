from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from neuralese_source_world_builder import build_proof, json_text, validate_world_rows
from neuralese_authored_worlds import iterate_world, reducer_world, result_event

OUT = ROOT / "runs/neuralese-successor-v14-prepared-20261007-candidate-v5-final-review"
REV = "authored-semantic-source-worlds-v14/1"


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
