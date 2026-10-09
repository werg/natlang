"""Sources (a) and (b) of refine-judge: the request/response plumbing around the teacher stages and the row assembly.

Flow (every arrow that needs the teacher runs in the teacher window, `plans/REFINEMENT_DATA.md`):

  harvest -> exemplify requests -> [teacher: exemplify] -> satisfying values
          -> near-miss requests -> [teacher: nearMiss] -> edited values
          -> verify requests    -> [teacher: verifyNearMiss] -> accepted pairs
  accepted pairs + mined pairs -> label requests -> [teacher judge: P(true)] -> rows

Rows are `natlang.decision-prompt/1` (common.judge_row). For near-miss rows `gold` is the construction's hard label (the
original satisfies, the edit does not; two passes agree) and `teachers[ID]` the teacher judge's soft distribution; for
mined rows there is no other label, so `gold` is the teacher's distribution. A near-miss pair on which the teacher judge
disagrees with the construction is not trained on; it is written to the disagreements list for review.
"""
from __future__ import annotations

from .common import canonical_json_str, judge_row, predicate_id, sha256_hex, split_of

NEAR_MISS_FAMILY = "refine-judge/near-miss"
MINED_FAMILY = "refine-judge/mined"


def _short(*parts) -> str:
    return sha256_hex("|".join(str(p) for p in parts).encode("utf-8"))[:12]


def exemplify_requests(candidates: list[dict], count: int = 6) -> list[dict]:
    """One request per distinct predicate (its first use supplies the base type and slot)."""
    rows = []
    for entry in candidates:
        use = entry["uses"][0]
        rows.append({"id": f"ex:{entry['predicate_id']}", "args": {"predicate": entry["predicate"], "base": use["base"],
                                                                   "slot": f"{use['app']}:{use['slots'][0]}", "count": count}})
    return rows


def satisfying_from_exemplify(requests: list[dict], results: list[dict]) -> list[dict]:
    by_id = {row["id"]: row for row in requests}
    out, seen = [], set()
    for result in results:
        request = by_id.get(result["id"])
        if not request or not result.get("ok"):
            continue
        values = (result.get("value") or {}).get("values")
        if not isinstance(values, list):
            continue
        predicate = request["args"]["predicate"]
        for value in values:
            key = (predicate_id(predicate), canonical_json_str(value))
            if key in seen or value is None or value == "":
                continue
            seen.add(key)
            out.append({"id": f"sat:{predicate_id(predicate)}:{_short(*key)}", "predicate": predicate, "value": value,
                        "origin": "exemplify"})
    return sorted(out, key=lambda row: row["id"])


def near_miss_requests(satisfying: list[dict]) -> list[dict]:
    return [{"id": "nm:" + row["id"], "args": {"predicate": row["predicate"], "value": row["value"]}} for row in satisfying]


def verify_requests(satisfying: list[dict], near_miss_results: list[dict]) -> list[dict]:
    by_id = {"nm:" + row["id"]: row for row in satisfying}
    out = []
    for result in near_miss_results:
        row = by_id.get(result["id"])
        edited = (result.get("value") or {}).get("edited") if result.get("ok") else None
        if row is None or edited is None:
            continue
        out.append({"id": "vf:" + row["id"], "args": {"predicate": row["predicate"], "original": row["value"], "edited": edited}})
    return out


def accept_near_misses(satisfying: list[dict], near_miss_results: list[dict], verify_results: list[dict]) -> tuple[list[dict], dict]:
    """Pairs both teacher passes support. Every rejection is counted by reason."""
    sat = {row["id"]: row for row in satisfying}
    edits = {r["id"][len("nm:"):]: r["value"] for r in near_miss_results if r.get("ok")}
    checks = {r["id"][len("vf:"):]: r["value"] for r in verify_results if r.get("ok")}
    stats: dict[str, int] = {}
    pairs = []
    for sat_id, row in sorted(sat.items()):
        def reject(reason):
            stats[reason] = stats.get(reason, 0) + 1
        edit, check = edits.get(sat_id), checks.get(sat_id)
        if edit is None:
            reject("no-edit")
        elif check is None:
            reject("no-verification")
        elif canonical_json_str(edit["edited"]) == canonical_json_str(row["value"]):
            reject("unchanged")
        elif type(edit["edited"]) is not type(row["value"]):
            reject("type-changed")
        elif check.get("original_holds") is not True:
            reject("verifier-original-fails")
        elif check.get("edited_holds") is not False:
            reject("verifier-edited-holds")
        elif check.get("minimal") is not True:
            reject("not-minimal")
        else:
            pairs.append({"id": sat_id, "predicate": row["predicate"], "original": row["value"], "edited": edit["edited"],
                          "edit": edit.get("edit", ""), "verifier_reason": check.get("reason", "")})
    stats["accepted"] = len(pairs)
    stats["considered"] = len(sat)
    return pairs, stats


def label_requests(accepted: list[dict], mined: list[dict]) -> list[dict]:
    """The pairs the teacher judge scores: both sides of every accepted near-miss pair, and every mined pair."""
    out = []
    for pair in accepted:
        out.append({"id": f"{pair['id']}:orig", "value": pair["original"], "predicate": pair["predicate"]})
        out.append({"id": f"{pair['id']}:edit", "value": pair["edited"], "predicate": pair["predicate"]})
    out += [{"id": pair["id"], "value": pair["value"], "predicate": pair["predicate"]} for pair in mined]
    return out


def teacher_labels(label_results: list[dict]) -> dict[str, dict[str, float]]:
    """id -> {teacher id: P(true)}, ignoring failed scorings."""
    out: dict[str, dict[str, float]] = {}
    for row in label_results:
        p = row.get("p_true")
        if isinstance(p, (int, float)) and 0.0 <= p <= 1.0:
            out.setdefault(row["id"], {})[row["teacher"]] = float(p)
    return out


def assemble(accepted: list[dict], mined: list[dict], label_results: list[dict], primary_teacher: str) -> tuple[list[dict], list[dict], dict]:
    """Rows, near-miss disagreements and a report. `primary_teacher` supplies the soft label that is checked and kept."""
    labels = teacher_labels(label_results)
    rows, disagreements = [], []
    report = {"near_miss_pairs": 0, "near_miss_rows": 0, "near_miss_disagreements": 0, "near_miss_unlabeled": 0,
              "mined_rows": 0, "mined_unlabeled": 0}

    def teachers_of(label_id):
        return {name: [p, 1.0 - p] for name, p in labels.get(label_id, {}).items()}

    for pair in accepted:
        report["near_miss_pairs"] += 1
        orig, edit = labels.get(f"{pair['id']}:orig", {}), labels.get(f"{pair['id']}:edit", {})
        if primary_teacher not in orig or primary_teacher not in edit:
            report["near_miss_unlabeled"] += 1
            continue
        if not (orig[primary_teacher] >= 0.5 and edit[primary_teacher] < 0.5):
            report["near_miss_disagreements"] += 1
            disagreements.append({**pair, "teacher": primary_teacher, "p_original": orig[primary_teacher], "p_edited": edit[primary_teacher]})
            continue
        split = split_of(pair["predicate"])
        for side, value, gold in (("orig", pair["original"], 1.0), ("edit", pair["edited"], 0.0)):
            rows.append(judge_row(row_id=f"refine-judge:near-miss:{pair['id']}:{side}", family=NEAR_MISS_FAMILY, split=split, value=value,
                                  predicate=pair["predicate"], gold_true=gold, teachers=teachers_of(f"{pair['id']}:{side}"),
                                  source="near-miss", label_source="construction+verification", edit=pair["edit"], pair_id=pair["id"]))
            report["near_miss_rows"] += 1
    for pair in mined:
        if primary_teacher not in labels.get(pair["id"], {}):
            report["mined_unlabeled"] += 1
            continue
        p = labels[pair["id"]][primary_teacher]
        rows.append(judge_row(row_id=f"refine-judge:{pair['id']}", family=MINED_FAMILY, split=split_of(pair["predicate"]),
                              value=pair["value"], predicate=pair["predicate"], gold_true=p, teachers=teachers_of(pair["id"]),
                              source="mined", label_source=f"teacher:{primary_teacher}", shadow_disagreement=bool(pair.get("shadow_disagreement")),
                              observed=pair.get("observed", [])[:3]))
        report["mined_rows"] += 1
    rows.sort(key=lambda row: row["id"])
    report["rows"] = len(rows)
    return rows, disagreements, report
