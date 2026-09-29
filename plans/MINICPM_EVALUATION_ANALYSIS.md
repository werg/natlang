# Sharp MiniCPM evaluation analysis — 2026-09-30

Model: Sharp-MiniCPM5-2B-Q4_K_XL, single seed42, temperature0, thinking enabled,
16k context,20 turns per call,120s per case and30-minute GPU reservation. This is
an incomplete baseline pilot, not a comparison of quantizations or decoding modes.

## Corrected accounting

24 held-out cases planned;22 attempted;11 final result records;11 timeout partials;
2 unattempted cases. Application probes were not reached. The original combined
results export was overwritten by each one-case collector invocation, producing a
false zero-collected report. Durable job recovery validates exact IR and provenance
hashes. Original reports/records remain intact.

- Original strict file checks:4 accepted /24 planned.
- JSON content rescore:5 accepted /24 planned; only the web release case changes.
- This rescores saved files; it makes no provider/GPU calls and is not action replay.
- Final job records include blocked/budget-exhausted outcomes;11 does not mean11 successful completions.

Evidence: `runs/minicpm-eval-20260929/report.recovered.json`,
`failure-analysis.json`, `file-content-rescore.json`, per-case jobs and events.
The failure analysis contains file hashes, per-case counts and observations.

## Failures with final records

| Case | What happened | Classification |
|---|---|---|
| Refund requiring detail (#1) | Helper incorrectly calls basic evidence sufficient, then interprets courier never visiting and parcel still at depot as delivery/fraud. Parent trusts this and returns fraud instead of legitimate. | Evidence interpretation and sufficiency mistake. Delegation itself works. |
| Receipt totals (#7) | Regex expects period decimals for comma-formatted EUR, and puts USD currency marker on wrong side. Ignores requested per-receipt judgment. Several runtime repairs followed by8 duplicate action sets; declares blocked though data is available. | Parsing, recovery and instruction-following failure. |
| Shipment aggregation (#9) | Starts with forbidden while; uses service call directly in finite-loop bound; ignores hint to cache bound. Invents `.pages` on numeric page count and misuses iterateOn.11 repeated action sets; blocked though query is computable. | Scope/API and recovery failure; restrictive loop surface adds friction. |
| Current roster (#10) | First tries in-place sort of read-only input, then copies it successfully. Deliberately ignores pending changes as “just context” and returns stale snapshot. | Semantic task misunderstanding; read-only error was recoverable. |
| Web release (#15) | Correct package selected, child bump invoked,1.8.9 →1.8.10, correct returned folder. Only file difference against expected is missing final newline. | Overly strict file oracle, fixed below. |
| Proof verifier (#17) | Repeats code redeclaring caller input `goal` despite explicit error,18 duplicate action sets, then invalid blocked return with value/no reason. Budget exhaustion. | Error-recovery and return-status failure. |
| Message triage (#18) | Reads complete input; customer-data location/EU-region question labeled setup instead of privacy. | Semantic classification mistake. |

## Timeout checkpoints

These were largely actively generating/repeating, not idle GPU hangs. Saved partials
contain108 response entries,100,983 completion tokens and625,867 prompt tokens
(prompt counts include cached context). About1,124s summed predicted decode time;
10/11 partial cases contain at least one length-capped response,17 caps total.
These counts include child calls and do not equal root turns. Nine of the partial
cases have obvious long reasoning cycles; one additional cap occurs in ticket repair.

- Release checks (#4,5), roster (#11), ticket filtering (#12), proof (#16) and
  deferral cases (#20,21): repeated or long self-correction, often exhausting the
  4,096-token response allowance without making a tool call.
- Receipts (#6) and shipments (#8): parsing/API/iteration confusion plus long reasoning.
- Ticket archive (#13): repetitive filesystem discovery/recovery.
- API release (#14): directory reducer confusion, repeated filesystem discovery;
  later model responses attempt edits to all packages despite only one having a
  code change, repeatedly bumping versions. Bash is supported here.41 responses
  include child work, not41 root turns.

Partial checkpoints do not retain full native tool outcomes. Actions and raw model
responses are observed; quoted errors, file edits and their success from partial
reasoning alone are not independently verified. They establish failure patterns,
not accepted results. The native/full-ledger cases above have stronger evidence.

## JSON whitespace fix

Default folder-result checks now compare `.json` files by strict parsed content:
object key order, indentation and trailing newline are irrelevant. Values, types,
array order, every field and string whitespace remain significant. Missing/extra
files, invalid JSON, duplicate keys and unsafe numeric values fail. Non-JSON files
retain exact text checks; byte-sensitive tasks can explicitly select `compare: exact`.
`compare: content` requires all files, irrespective of a partial-credit threshold.
Formatting-only JSON writes do not count as semantic changes under that oracle.

The saved web-release result passes all four files under the new comparison; other
saved verdicts do not change. New collector provenance/reuse keys record
`file_content_comparison_version: json-content/1`, preventing obsolete strict verdict
reuse. Existing positive byte-exact evidence remains valid, so the broad data-quality
version is not bumped and previously reviewed corpora are not needlessly excluded.

Built current Node runtime; immutable runtime ready for subsequent campaigns:
`runs/file-content-oracle-20260930/runtime-v1`. Active Bonsai v39 and Luna visible v4
remain frozen; do not mutate their files. Current pipeline freezes the updated build
for future runs. Completed evaluation has been rescored without rerunning the GPU.

## What to address next

1. Recovery training: modify the exact failing line after a diagnostic, cache a loop
   bound, use injected inputs directly, copy read-only arrays, return valid status.
2. Semantic training: snapshot plus ordered deltas, evidence sufficiency, final paid
   total versus line items, topic disambiguation and selective file edits.
3. Runtime usability: review finite-loop-bound handling and diagnostic specificity;
   do not remove bounded execution or weaken actual content checks.
4. Decoding comparison: temperature0 versus a sampled configuration and shorter
   reasoning budgets, same case/seed accounting, saved per-case exports. Current
   evidence cannot attribute looping to greedy decoding, template, quantization or
   underlying model independently. Tool parsing and delegation worked in several
   cases, so there is no evidence of a wholesale template/tool-call failure.
5. Finish missing cases and reserve a separate bounded application-probe slot.

Keep these held-out inputs/trajectories excluded from training. Cover patterns with
new independent training variants and preserve source-group separation.
