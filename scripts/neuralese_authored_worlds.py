"""Reusable constructors for authored semantic source worlds.

Source facts are supplied explicitly by each campaign builder; these helpers
only package nested folder reducers and ordered complete-draft iteration rows.
"""
from __future__ import annotations

import json
import re
from neuralese_source_world_builder import json_text

DEFAULT_REVISION = "authored-semantic-source-worlds-v14/1"

def result_event(value):
    return ["return_result", {"status": "success", "value": value}]


def source_version(revision: str) -> int:
    match = re.search(r"source-worlds-v(\d+)", revision)
    if not match:
        raise ValueError(f"unsupported source revision {revision}")
    return int(match.group(1))


def reducer_world(slug, split, domain, inherited, teams, records, clauses, item_scope, revision=DEFAULT_REVISION):
    version = source_version(revision)
    family = f"authored_semantic_source_worlds_v{version}"
    group = f"v{version}:{slug}:world"
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
            clause_audit.append({"id": record["id"], "team": record.get("team"),
                                 "thing": record.get("thing"), "site": record.get("site"),
                                 "text": record.get("text"), "authority_evidence": record.get("authority_evidence"),
                                 "clauses": record["supports"],
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
        "id": f"inline-curriculum:{revision.replace('/', '_')}:v{version}-{slug}-nested-reducer:inherited-local-item-reconciliation",
        "kind": "lambda_source", "family": f"curriculum_{family}",
        "source": "natlang-inline-curriculum", "split": split,
        "source_ids": [group], "source_groups": [group], "source_revisions": [revision],
        "license": "project-generated", "gold_sources": ["constructed-world-oracle"],
        "generation": {
            "generator": revision, "independent_world": slug,
            "capture_contract": {"inherited_rule": "folder policy JSON", "local_rule": "same-folder override JSON",
                                 "item_facts": "same-item FileHandle", "cross_item_evidence": "forbidden"},
            "source_quality": "each negative item fails exactly one explicit applicable clause",
        },
        "curriculum": {
            "version": "natlang.inline_curriculum/1", "family": family,
            "family_version": version, "shape": f"v{version}-{slug}-nested-reducer",
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


def iterate_world(slug, split, domain, fields, passes, initial, packet, revision=DEFAULT_REVISION):
    version = source_version(revision)
    family = f"authored_semantic_source_worlds_v{version}"
    group = f"v{version}:{slug}:world"
    keys = [field for field, _, _ in fields]
    shape = "{ " + "; ".join(f"{field}: string" for field in keys) + " }"
    pass_json = json_text([{"name": name, "constraint": constraint} for name, constraint, _ in passes])
    def exact_field_contract(contract: str) -> str:
        if "only the exact" in contract.casefold() and "token" in contract.casefold():
            return contract
        return contract.rstrip(".") + "; copy the exact source wording, preserving its spelling and punctuation, without paraphrasing or adding explanation."

    task = {
        "instruction": f"Reconcile the {domain} record using the ordered source checks.",
        "output_path": "decision.json",
        "output_contract": {
            "format": "JSON object with exactly the declared string fields.",
            "fields": {field: exact_field_contract(contract) for field, _, contract in fields},
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
        "id": f"inline-curriculum:{revision.replace('/', '_')}:v{version}-{slug}-iterate:ordered-complete-draft-reconciliation",
        "kind": "lambda_source", "family": f"curriculum_{family}",
        "source": "natlang-inline-curriculum", "split": split,
        "source_ids": [group], "source_groups": [group], "source_revisions": [revision],
        "license": "project-generated", "gold_sources": ["constructed-world-oracle"],
        "generation": {
            "generator": revision, "independent_world": slug,
            "capture_contract": {"instruction": "task-visible", "contract": "exact declared fields", "pass": "ordered current-pass only",
                                 "current_draft": "complete actual carried draft", "evidence": "same packet FileHandle"},
            "source_quality": "each pass changes one assigned field from explicit current-pass evidence and carries the full draft",
        },
        "curriculum": {
            "version": "natlang.inline_curriculum/1", "family": family,
            "family_version": version, "shape": f"v{version}-{slug}-iterate", "variant": "ordered-complete-draft-reconciliation",
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
