#!/usr/bin/env python3
"""Freeze semantic follow-up cases for the native teacher collector.

The report is hidden behind a callable. Its classification depends on the
meaning of the prose, including negation and conflicting surface cues.
"""
import argparse
import json
from pathlib import Path


SCENARIOS = {
    "incident": {
        "labels": ("security", "capacity", "dependency"),
        "policy": "Route a confirmed exposure or unauthorized access to security; resource exhaustion to capacity; and a failure caused by another service to dependency. A ruled-out possibility is not the cause.",
        "reports": [
            ("security", "The checkout API is responsive. Audit logs show a token belonging to another tenant read private invoices; the CPU alarm was a stale dashboard."),
            ("capacity", "Users see timeouts only during traffic peaks. Workers are saturated and recover as load falls; the security scanner found no unauthorized access."),
            ("dependency", "Our queue depth and CPU are normal. Requests fail because the upstream tax service rejects every connection after its deployment."),
            ("security", "A cache miss raised latency, but the urgent finding is that an unauthenticated request can fetch another customer's profile."),
            ("capacity", "An alert mentions credentials, but the credentials are valid. The image service runs out of memory while processing larger uploads."),
            ("dependency", "The status page says degraded capacity, yet local workers are idle and the external payment gateway returns 503 for each charge."),
            ("security", "The public error page is stable. A leaked signing key was used to mint valid sessions from an unfamiliar network."),
            ("capacity", "The database is healthy and authorization checks pass. Disk space reaches zero whenever the batch importer starts."),
            ("dependency", "Retrying locally changes nothing: the identity provider has stopped issuing tokens, although our authentication code and CPU are healthy."),
            ("security", "The burst of requests resembles load trouble, but sampled requests are reading files outside the requesting account's scope."),
            ("capacity", "The vendor outage is an old resolved ticket. Current requests queue behind exhausted connection pool slots in our own service."),
            ("dependency", "A security test failed in staging only. Production failures began when the DNS provider stopped resolving the partner endpoint."),
        ],
    },
    "customer": {
        "labels": ("refund", "replacement", "clarify"),
        "policy": "Choose refund when the customer clearly asks to cancel or get money back; replacement when they want the purchased item fulfilled again; clarify when intent or eligibility is genuinely unresolved. Do not treat a quoted or rejected option as their request.",
        "reports": [
            ("refund", "The box arrived damaged. I do not want another unit; please return the payment to my card."),
            ("replacement", "The screen is cracked. I still need this model for work and would like an intact one sent out."),
            ("clarify", "The parcel may be at the neighbor's house, and I haven't checked yet. What would happen if it is missing?"),
            ("refund", "Support offered a replacement, but the event is over now. Please cancel the order and reverse the charge."),
            ("replacement", "A refund was mentioned in the earlier email. After checking stock, I prefer the same size shipped again."),
            ("clarify", "I wrote 'refund' in the subject because that was the menu choice. I am still deciding whether to keep the item."),
            ("refund", "The delivered item works, but it arrived after the trip. I have no use for a replacement and request my money back."),
            ("replacement", "One part is missing from the kit. I want the complete kit, not a credit or cancellation."),
            ("clarify", "I can see two charges and one shipment. Before choosing a remedy, can someone explain whether the second charge is pending?"),
            ("refund", "Please do not reship. The recipient moved away and I want this purchase voided."),
            ("replacement", "The original was lost by the courier. The event is next month, so a new shipment is still useful."),
            ("clarify", "The tracking page says delivered, but the office may have accepted it. I will check with reception and update you."),
        ],
    },
    "evidence": {
        "labels": ("supports", "contradicts", "inconclusive"),
        "policy": "Judge whether the observation supports, contradicts, or leaves the claim unresolved. Match the population and time period; a related metric or an unrepresentative sample is not proof.",
        "reports": [
            ("supports", "Claim: the new cache reduces median lookup latency. A controlled comparison on the same workload measured 82 ms before and 51 ms after, with unchanged hardware."),
            ("contradicts", "Claim: the retry change lowers duplicate payments. In matched production windows, duplicates rose from 3 to 19 after rollout."),
            ("inconclusive", "Claim: the new parser handles all international addresses. The test used only US addresses and passed every one."),
            ("supports", "Claim: disabling the batch job frees memory at night. Memory stayed near 90% on nights with the job and near 42% on otherwise comparable nights without it."),
            ("contradicts", "Claim: the migration preserves every account id. The reconciliation lists six ids present before migration and absent afterward."),
            ("inconclusive", "Claim: the endpoint is faster for mobile users. Desktop timings improved, but no mobile requests were measured."),
            ("supports", "Claim: requests are rejected after a revoked key is used. A replay with the revoked key returned 401; the same request with a valid key returned 200."),
            ("contradicts", "Claim: no private fields reach logs. A sampled production log line contains a full customer access token."),
            ("inconclusive", "Claim: the change fixes intermittent outages. One hour passed without an outage, but historical gaps between outages are often several days."),
            ("supports", "Claim: the new index cuts full table scans on this query. Query plans from the same database before and after show a scan first and an indexed lookup after."),
            ("contradicts", "Claim: every tenant was notified. Delivery records show that two active tenants have no notification entry."),
            ("inconclusive", "Claim: the bug only affects older browsers. The sole reproduction was on an old browser; newer browsers were never tried."),
        ],
    },
    "code_diagnosis": {
        "labels": ("implementation", "contract", "test_expectation"),
        "policy": "Classify the most direct cause: implementation logic is wrong, caller and callee disagree on an interface contract, or the test expects behavior contrary to the stated specification. Do not infer from an error name alone.",
        "reports": [
            ("implementation", "Spec: keep first occurrence order. The function constructs a Set, then sorts its values alphabetically; the failing test expects original order."),
            ("contract", "The helper now returns {items, cursor}; the caller still treats its return as an array and invokes .map on the entire object."),
            ("test_expectation", "Spec: zero is a valid threshold. The test insists threshold zero must throw, although the function correctly returns an empty selection."),
            ("implementation", "The loop skips index zero and never accounts for it, even though the spec says sum every element."),
            ("contract", "The API documents milliseconds, while the caller supplies seconds. Both computations are internally consistent but disagree at the boundary."),
            ("test_expectation", "Spec requires stable sorting for equal scores. The implementation preserves input order; the test expects equal-score ids alphabetized."),
            ("implementation", "The function compares names with case sensitivity although the spec explicitly says comparisons ignore case."),
            ("contract", "A renamed field is now 'accountId' in the producer, but the consumer reads 'account_id' and gets undefined."),
            ("test_expectation", "Spec allows an empty input and requires an empty output. The test expects a sentinel object on empty input."),
            ("implementation", "The retry counter resets inside the loop, so the promised maximum of three attempts is never enforced."),
            ("contract", "The callee returns a Promise of a record, while the caller reads the record's fields without awaiting it."),
            ("test_expectation", "Spec defines the boundary as inclusive. The implementation includes the endpoint, but the assertion expects the endpoint omitted."),
        ],
    },
    "deployment": {
        "labels": ("proceed", "hold", "rollback"),
        "policy": "Proceed when evidence supports continuing a safe rollout, hold when evidence is incomplete or a pre-launch blocker remains, and roll back when the active rollout causes a material regression. Distinguish a current regression from an unrelated historical alert.",
        "reports": [
            ("proceed", "The canary has served representative traffic for two hours. Error rate and latency match baseline, and the required migration check passed."),
            ("hold", "The canary is healthy, but the required data migration has not run and no rollback snapshot exists yet."),
            ("rollback", "Since the active rollout began, checkout errors rose from 0.2% to 8% on the canary while controls stayed flat."),
            ("proceed", "An old alarm remains red in the dashboard, but it predates this release. Fresh canary and control metrics are both healthy."),
            ("hold", "Only three synthetic requests have reached the canary. There is no production traffic evidence yet for the changed endpoint."),
            ("rollback", "The rollout changed the authorization layer. New canary requests can read another tenant's records; controls cannot."),
            ("proceed", "The release gate passed, canary metrics are stable, and the one observed retry spike also occurred in the control group."),
            ("hold", "A required dependency version is absent from the package lock. The canary has not exercised the affected path."),
            ("rollback", "After enabling the release, writes from canary instances began failing persistently; equivalent control writes succeed."),
            ("proceed", "The preflight check and backup completed. Representative canary traffic shows no meaningful change in failures or p95 latency."),
            ("hold", "Metrics look normal, but the canary has been running for only one minute and the scheduled security check is pending."),
            ("rollback", "The new version is live and causes duplicate invoices on real requests. Disabling its flag stops the duplicates."),
        ],
    },
}

# Each tuple has two independently measured populations plus a prose qualification.
# The reference features are retained for analysis; they are not shown to the model.
COMPOSITES = {
    "rollout": {
        "policy": "Compare the change in canary error rate with the change in control. A material canary-only regression linked to this release calls for rollback; incomplete attribution or an outstanding gate calls for hold; stable or improving evidence with gates complete supports proceeding.",
        "rows": [
            ([1, 1, 2], [1, 2, 1], [1, 2, 1], [9, 10, 8], "The new parser is the only canary change. Replaying the old request through it reproduces the errors; reverting it stops them.", "regressed", "linked", "rollback"),
            ([2, 2, 1], [2, 1, 2], [2, 1, 2], [2, 2, 1], "All release gates passed. A headline incident is from yesterday's retired build and has no current requests.", "stable", "unrelated", "proceed"),
            ([1, 2, 1], [6, 5, 6], [1, 1, 2], [6, 6, 5], "The provider outage affected canary and control equally; the canary's changed path passed its checks.", "stable", "unrelated", "proceed"),
            ([1, 2, 1], [1, 1, 2], [2, 1, 1], [7, 8, 7], "A new failure is visible, but the trace IDs were lost and no one can yet attribute it to this release or the upstream.", "regressed", "uncertain", "hold"),
            ([5, 6, 5], [5, 5, 6], [6, 5, 6], [2, 1, 2], "Canary traffic exercises the changed path, required checks passed, and there is no unresolved defect report.", "improved", "linked", "proceed"),
            ([1, 1, 2], [1, 1, 2], [2, 1, 2], [2, 2, 1], "Metrics are steady, but the mandatory backup snapshot has not completed. This gate must precede wider rollout.", "stable", "blocker", "hold"),
            ([2, 1, 2], [2, 2, 1], [1, 2, 1], [10, 9, 10], "Only canary nodes use the new authorization check. A request to another tenant succeeds there and fails in control.", "regressed", "linked", "rollback"),
            ([4, 5, 4], [8, 9, 8], [4, 4, 5], [8, 8, 9], "The shared database had an incident; both populations moved together and the changed canary code was not on the failing path.", "stable", "unrelated", "proceed"),
            ([1, 2, 1], [1, 2, 1], [2, 2, 1], [5, 6, 5], "The canary worsened, but its traffic mix shifted to a new region at the same time. No matched comparison is available yet.", "regressed", "uncertain", "hold"),
            ([7, 8, 7], [7, 7, 8], [8, 7, 8], [3, 2, 3], "The release removes a known retry storm; replay verifies that the same request no longer loops.", "improved", "linked", "proceed"),
            ([1, 2, 1], [1, 1, 2], [1, 2, 1], [1, 2, 1], "The canary is healthy, but the required security sign-off is still pending and cannot be waived by good metrics.", "stable", "blocker", "hold"),
            ([2, 1, 2], [2, 2, 1], [2, 1, 2], [8, 9, 8], "The new serializer corrupts persisted records on canary writes. Control writes remain intact; the fault disappears with the release flag off.", "regressed", "linked", "rollback"),
        ],
    },
    "study": {
        "policy": "Compare treatment change with control change, then judge whether the study note makes that comparison credible. A credible favorable effect supports the claim, a credible adverse effect contradicts it, and a confounded or unmeasured effect is inconclusive.",
        "rows": [
            ([40, 41, 39], [41, 40, 41], [40, 39, 41], [50, 51, 49], "Assignment was randomized, groups used the same measurement window, and the outcome was specified before enrollment.", "positive", "credible", "supports"),
            ([40, 39, 41], [40, 41, 39], [41, 40, 39], [32, 31, 33], "The allocation and outcome logging were unchanged, with complete observations for both groups.", "negative", "credible", "contradicts"),
            ([40, 41, 39], [40, 39, 41], [40, 39, 41], [50, 49, 51], "Treatment users were recruited from a premium campaign, while controls were sampled from ordinary traffic; the groups were never matched.", "positive", "confounded", "inconclusive"),
            ([40, 39, 41], [44, 45, 43], [40, 41, 39], [44, 43, 45], "Both groups faced the same seasonal demand shift. The outcome was recorded for every assigned user.", "neutral", "credible", "inconclusive"),
            ([50, 49, 51], [50, 51, 49], [50, 49, 51], [43, 42, 44], "The adverse movement appeared only after treatment assignment; instrumentation and cohorts remained comparable.", "negative", "credible", "contradicts"),
            ([30, 31, 29], [30, 29, 31], [30, 29, 31], [41, 40, 42], "Half of the treatment outcomes are missing because the new client stopped sending telemetry. Missingness is not random.", "positive", "confounded", "inconclusive"),
            ([55, 54, 56], [60, 59, 61], [55, 56, 54], [60, 61, 59], "The treatment and control moved together in the same window; assignment was randomized.", "neutral", "credible", "inconclusive"),
            ([20, 21, 19], [20, 19, 21], [20, 19, 21], [27, 28, 26], "The comparison was preregistered, assignment was random, and neither arm changed its logging method.", "positive", "credible", "supports"),
            ([45, 44, 46], [45, 46, 44], [45, 44, 46], [37, 38, 36], "The treatment arm was run in a different country during a holiday; control traffic came from the original region.", "negative", "confounded", "inconclusive"),
            ([60, 61, 59], [60, 59, 61], [60, 59, 61], [69, 70, 68], "No assignment audit exists, so the apparent lift could come from the enrollment process rather than the treatment.", "positive", "unknown", "inconclusive"),
            ([40, 39, 41], [40, 41, 39], [40, 41, 39], [31, 32, 30], "Randomization and matched observation windows were verified; no outcome records were dropped.", "negative", "credible", "contradicts"),
            ([35, 34, 36], [35, 36, 34], [35, 34, 36], [35, 36, 34], "The sample is small but complete and randomized. There is no directional movement to attribute.", "neutral", "credible", "inconclusive"),
        ],
    },
    "fault_locality": {
        "policy": "Compute how local and upstream failure counts changed, then use trace evidence to identify the fault. A simultaneous increase alone does not prove which system caused it. Choose local, upstream, or unresolved.",
        "rows": [
            ([1, 2, 1], [1, 1, 2], [1, 1, 2], [9, 8, 10], "Local stack traces point to the new decoder before any upstream request is sent.", "local_only", "points_local", "local"),
            ([1, 1, 2], [9, 10, 8], [1, 2, 1], [8, 9, 10], "Requests reach the dependency; its logs show connection refusal before our service reports an error.", "both", "points_upstream", "upstream"),
            ([1, 2, 1], [1, 2, 1], [2, 1, 2], [8, 9, 8], "A newly deployed local validation rule rejects requests without calling any dependency.", "local_only", "points_local", "local"),
            ([1, 2, 1], [9, 8, 10], [1, 2, 1], [1, 1, 2], "The partner service changed its certificate. Its handshake fails before our handler receives a response.", "upstream_only", "points_upstream", "upstream"),
            ([1, 2, 1], [8, 9, 8], [2, 1, 2], [8, 9, 8], "Both logs show failures, but trace propagation is broken, so their order and cause are unknown.", "both", "ambiguous", "unresolved"),
            ([1, 1, 2], [2, 1, 1], [1, 2, 1], [2, 1, 1], "A customer supplied a screenshot from last month; there is no current reproduced failure.", "neither", "ambiguous", "unresolved"),
            ([2, 2, 1], [10, 9, 8], [1, 1, 2], [10, 9, 8], "The dependency succeeds when called directly. Local code truncates its response and then throws.", "both", "points_local", "local"),
            ([1, 1, 2], [8, 9, 10], [1, 2, 1], [1, 2, 1], "The upstream reports exhausted workers and sends 503; local retries merely surface those failures.", "upstream_only", "points_upstream", "upstream"),
            ([1, 2, 1], [1, 1, 2], [1, 1, 2], [9, 9, 8], "The local parser now rejects valid payloads. Upstream responses and timings remain healthy.", "local_only", "points_local", "local"),
            ([1, 2, 1], [9, 8, 10], [1, 2, 1], [8, 10, 9], "An upstream timeout is observed, but the local deploy also changed retry policy; traces cannot establish which initiated the incident.", "both", "ambiguous", "unresolved"),
            ([2, 1, 1], [8, 10, 9], [1, 2, 1], [1, 1, 2], "Dependency health probes fail independently of our calls, and its operators report an outage.", "upstream_only", "points_upstream", "upstream"),
            ([1, 2, 1], [8, 9, 10], [2, 1, 1], [8, 9, 10], "The first failing span is in local request assembly; the dependency sees malformed calls but otherwise remains healthy.", "both", "points_local", "local"),
        ],
    },
}


def make_record(family, index, category, report, policy, labels):
    case_id = f"{family}-{index:02d}"
    reports = {f"{family}-{i:02d}": text for i, (_, text) in enumerate(SCENARIOS[family]["reports"])}
    codebase = {
        "lookup_report": {
            "description": "Retrieve the case report by id.",
            "args": {"case_id": "string"}, "returns": "string",
            "code": f"const reports: Record<string, string> = {json.dumps(reports)};\nreturn reports[case_id];",
            "async": False,
        },
        "record_decision": {
            "description": "Record a category chosen after interpreting the report.",
            "args": {"case_id": "string", "category": "string"}, "returns": "Decision",
            "code": "return { case_id, category };",
            "types": {"Decision": "{ case_id: string, category: string }"},
            "async": False,
        },
    }
    instructions = (
        "Get the case report with lookup_report(case_id) and inspect what it says.\n"
        f"Interpret the whole report using this policy: {policy} The categories are {', '.join(labels)}.\n"
        "Record the category you chose with record_decision and return its Decision."
    )
    return {
        "version": "natlang.program/1", "id": f"semantic-followup:{case_id}",
        "kind": "lambda_source", "family": f"semantic_followup_{family}",
        "source": "natlang-semantic-followup-probe", "split": "test",
        "source_ids": [case_id], "source_groups": [case_id],
        "source_revisions": ["natlang.semantic_followup_probe/1"],
        "license": "project-generated", "gold_sources": ["hand-reviewed-semantic-scenarios"],
        "generation": {"generator": "natlang.semantic_followup_probe/1", "index": index,
                       "report": report, "reference_category": category},
        "semantics": {
            "root": {"$lambda": {
                "type": "(case_id: string) => Decision",
                "types": {"Decision": "{ case_id: string, category: string }"},
                "instructions": instructions, "function": f"decide_{family}",
                "codebase": codebase,
            }},
            "inputs": {"case_id": case_id},
            "expected": {"case_id": case_id, "category": category},
            "operation": "semantic_followup",
        },
    }


def make_composite_record(family, index, row, policy):
    control_before, control_after, focus_before, focus_after, note, trend, context, conclusion = row
    case_id = f"{family}-{index:02d}"
    notes = {f"{family}-{i:02d}": item[4] for i, item in enumerate(COMPOSITES[family]["rows"])}
    fields = {
        "case_id": case_id, "trend": trend, "context": context, "conclusion": conclusion,
    }
    vocab = {field: sorted({item[position] for item in COMPOSITES[family]["rows"]})
             for field, position in (("trend", 5), ("context", 6), ("conclusion", 7))}
    if family == "fault_locality":
        type_signature = "(case_id: string, upstream_before: number[], upstream_after: number[], local_before: number[], local_after: number[]) => Assessment"
        inputs = {"case_id": case_id, "upstream_before": control_before,
                  "upstream_after": control_after, "local_before": focus_before,
                  "local_after": focus_after}
        measure_lines = (
            "Compute the before/after averages separately for upstream failures and local failures.\n"
            "Determine whether each average increased by more than 2; record the combined pattern as upstream_only, local_only, both, or neither.\n"
        )
    else:
        type_signature = "(case_id: string, control_before: number[], control_after: number[], focus_before: number[], focus_after: number[]) => Assessment"
        inputs = {"case_id": case_id, "control_before": control_before,
                  "control_after": control_after, "focus_before": focus_before,
                  "focus_after": focus_after}
        measure_lines = (
            "Compute the control before/after averages and the focus before/after averages.\n"
            "Compare their changes to infer the adjusted trend; treat an adjusted difference with magnitude at most 2 as stable or neutral.\n"
        )
    return {
        "version": "natlang.program/1", "id": f"semantic-followup:{case_id}",
        "kind": "lambda_source", "family": f"semantic_followup_{family}",
        "source": "natlang-semantic-followup-probe", "split": "test",
        "source_ids": [case_id], "source_groups": [case_id],
        "source_revisions": ["natlang.semantic_followup_probe/1"],
        "license": "project-generated", "gold_sources": ["hand-reviewed-semantic-scenarios"],
        "generation": {"generator": "natlang.semantic_followup_probe/1", "index": index,
                       "note": note, "reference_features": fields},
        "semantics": {
            "root": {"$lambda": {
                "type": type_signature,
                "types": {"Assessment": "{ case_id: string, trend: string, context: string, conclusion: string }"},
                "instructions": (
                    measure_lines +
                    "Get the investigation note with lookup_context(case_id) and interpret its meaning.\n"
                    f"Combine the computed trend and the note: {policy} "
                    f"Use these exact labels: trend = {', '.join(vocab['trend'])}; "
                    f"context = {', '.join(vocab['context'])}; "
                    f"conclusion = {', '.join(vocab['conclusion'])}.\n"
                    "Record the trend, context finding, and conclusion with record_assessment; return its Assessment."
                ),
                "function": f"assess_{family}",
                "codebase": {
                    "lookup_context": {
                        "description": "Retrieve the independent investigation note for this case.",
                        "args": {"case_id": "string"}, "returns": "string",
                        "code": f"const notes: Record<string, string> = {json.dumps(notes)};\nreturn notes[case_id];",
                        "async": False,
                    },
                    "record_assessment": {
                        "description": "Record a semantic assessment after computing and interpreting its evidence.",
                        "args": {"case_id": "string", "trend": "string", "context": "string", "conclusion": "string"},
                        "returns": "Assessment",
                        "types": {"Assessment": "{ case_id: string, trend: string, context: string, conclusion: string }"},
                        "code": "return { case_id, trend, context, conclusion };",
                        "async": False,
                    },
                },
            }},
            "inputs": inputs,
            "expected": fields,
            "operation": "semantic_composite_followup",
        },
    }
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/teacher/semantic-followup-probe.ir.jsonl"))
    args = parser.parse_args()
    rows = [make_record(family, i, category, report, spec["policy"], spec["labels"])
            for family, spec in SCENARIOS.items() for i, (category, report) in enumerate(spec["reports"])]
    rows += [make_composite_record(family, i, row, spec["policy"])
             for family, spec in COMPOSITES.items() for i, row in enumerate(spec["rows"])]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows))
    print(f"{len(rows)} cases across {len(SCENARIOS) + len(COMPOSITES)} semantic families -> {args.out}")


if __name__ == "__main__":
    main()
