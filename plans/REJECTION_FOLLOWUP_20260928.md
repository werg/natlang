# Rejection follow-up — 2026-09-28 evening

Evidence: `runs/rejection-followup-20260928-evening/`. Original jobs/journals
remain intact. No training started. Snapshot counts: 78 v10 final results,
Bonsai 31 accepted / 20 rejected, Luna 5 accepted / 22 rejected; seven further
attempts had no final result and remain incomplete evidence, not negatives.

## Findings and fixes

- Two correct recursion repairs were falsely held: replacing the planted first
  compile failure prevented the error whose observation admission demanded.
  Collector/admission now share the replacement predicate. Only compile-error
  observations are waived, with matching root action, empty prefix and collector
  evidence. Runtime/state observations remain required.
- Old board tasks exposed application state as editable program source. Models
  removed the audit lock or repeated committed moves after a view failure. Those
  results remain rejected. Migration exposes the board as an external service.
  Eval errors now distinguish discarded new bindings from persistent live/module/
  file/service state and ask agents to inspect state before retrying writes.
- Mail had no readable declaration and an undocumented successful null return.
  `read_code("mail")` now explains that null confirms delivery and identical keyed
  retries are safe. Malformed commands and changed payloads under a delivered key
  throw before another delivery.
- Both supplied highlighter `matchAll` loops now spread finite matches before
  iteration. Native iterator restrictions were retained; old task snapshots get
  the same repair.
- CommaQA v2 explains local questions, passing names across stores, complete
  pagination and measurement strings. Reference eval now computes arithmetic and
  uses prior returned values, instead of copying gold intermediates/final numbers.
  All 25,098 numeric steps across train/dev/test agree with the compiler. This
  broader audit caught our initial signed subtraction: the source means absolute
  difference. Corrected before rollout; no source labels changed.
- ALFWorld request budgets were larger but the supervisor hardcoded 20 turns.
  Entries now accept validated `max_turns`; world retries get 48 and at least
  1,200 seconds. Existing request/concurrency caps remain bounded.
- Six αNLI inputs are held for source adjudication: their visible alternatives
  are insufficiently distinguished by the endpoints. Original labels remain.
  The shared registry catches legacy batches by complete visible identity.
  Fresh batches record every source identity and story group, rather than only
  the first story's group. Other label disagreements remain rejected.

## Breaking decisions

1. CommaQA numeric family versions below 2 are quarantined from collection,
   admission and pairs until migrated/replayed. Earlier ready artifacts containing
   those contracts or the six disputed αNLI inputs must be rebuilt with current
   gates; earlier static counts are historical.
2. Changed snapshots get new `:contracts-v2` identities, preserving source IDs/
   groups. Incompatible handoff prefixes restart from the root; resets and original
   identities are logged in `migration.jsonl`. Raw histories are never rewritten
   into successes.
3. Changed contents under a delivered mail key now fail rather than silently
   succeeding. Successful sends still return null for compatibility.

## Rollout and verification

Prepared Bonsai `runs/bonsai-recovery/queue-v8.jsonl` and Luna
`runs/luna-repair-20260928/queue-v4.jsonl`. New v11 jobs have distinct paths;
changed curriculum references pass native replay. Infrastructure/contract retries
are prioritized; pending source reviews are excluded. Preparation counts live in
`migration-summary.json`; already journaled finishes are skipped on start.
V10 stays immutable; v11 rollout audit/state records actual boundaries and PIDs.
Exactly one Luna worker/request remains authorized. Bonsai continues independently.

Verification: Node/browser builds, browser types, 109 focused Node tests, two queue
tests, full 25,098-step numeric source audit and changed-reference replays.
Correct direct/delegated/mixed outcomes remain eligible. The three ad hoc NL-layer
limit is unchanged. Remaining folder criteria errors retain their original gold.

## Later results: FOLIO source-language mismatches

`later-results.json` records results arriving during this audit. FOLIO story 337
uses an English disjunction permitting Jim to be a fast professional basketball
player; its source formalization instead uses negated XOR of positive predicates.
That changes the conclusions. Story 162 formalizes an invitation as actual concert
performance. Both stories are now held across single-conclusion and batch adapters,
using source groups or legacy shapes. Raw English, formalizations and labels remain.
This is a source hold, not a model error or relabeling.

Luna reached its v10 boundary and began its highlighter retry on v11. Added holds
require a new immutable v12 freeze; do not mutate v11. Bonsai queue-v9 and Luna
queue-v5 prune affected stories from the reviewed v11 queues and use distinct v12
job paths. The v12 rollout monitor waits for progressing cases to finish and records
new supervisor PIDs in `runs/bonsai-recovery/runtime-v12.rollout-state.json`.
Latest verification: 110 focused Node tests, Node/browser builds and browser types.
