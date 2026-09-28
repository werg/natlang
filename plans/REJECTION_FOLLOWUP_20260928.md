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

## Export protection and highlighter oracle review

Direct native materialization now applies source holds and retired-contract policy,
including failed-run exports. This closes a bypass around the admission CLI.
Validation: 127 focused Node checks, browser builds/types. Static reference reaudit
replayed all 2,192 integrated references: all still admitted, 5,174 approved / 5,054
held decisions before final dedup. These references contain none of the newly held
source inputs; their earlier audit counts did not shrink.

The v11 highlighter retry fixed iterator execution but exposed a stale oracle:
the seed labels map/for-each function calls as `call` despite the role definitions,
labels literal `{}` assignment as prose, and expects CSS whitespace different from
its supplied renderer. No repaired teacher result was admitted. Hold all legacy
`cb_highlighter` oracles until independently rebuilt (`highlighter_quality_version: 2`).
The current-source freezer v3 now excludes held cases and writes a rejection ledger:
36 eligible cases in nine codebases, four highlighter seeds held. It cannot silently
carry old gold into a new source snapshot.

Generation rollout v13 includes these gates; latest queues are Bonsai queue-v10 and
Luna queue-v6. Source shapes/groups and previous jobs remain immutable. The already
quarantined, repetitively failing Bonsai handoff 23 is stopped with a source-review
event rather than a training-negative verdict; this exception avoids spending its
remaining budget on data we cannot use. Valid progressing cases retain boundary
migration. First expanded ALFWorld retry succeeded: score 100, 22 moves.

Live checkpoint: Bonsai supervisor 395860 runs v13 / queue-v10. Its first repaired
locked-board case is accepted as an honest blocked outcome. Luna supervisor 393617
runs v12 / queue-v5 with one worker/request: first two world retries are accepted
at score 100 (22 and 58 moves). Monitor 395950 waits for its current valid case's
boundary, then starts v13 / queue-v6; actual completion is recorded in the rollout
log/state. `repair-progress.json` preserves this checkpoint. These are early repair
successes, not a measured success-rate claim.

## Later check-in: numeric evidence corruption (runtime v14)

Raw v13 snapshot (`runs/rejection-checkin-20260928-late/index.json`): Bonsai
10 accepted / 14 rejected final results, plus one timeout; Luna one accepted /
two rejected final results, plus four incomplete attempts and one timeout. These
are attempt counts, not unique cases or a quality-adjusted success rate. Mail
retries passed 4/4; board retries passed 3/4. Several rejected attempts concern
the same disputed source input.

**Course correction:** the earlier 25,098-step arithmetic audit was insufficient.
CommaQA numeric's upstream nationality table templates swap discus and javelin,
while KB predicates, language configuration, QA facts and throw passages agree.
The new evidence audit checks all splits: 64,000 table facts, all 64,000 original
renderings inverted; corrected table renderings and original throw evidence have
zero audit failures. Raw upstream data and answer labels remain unchanged.
Canonical table sentences now derive from `table_nationd` (discus) and
`table_nationj` (javelin), with repair provenance. Numeric contract versions 1/2
are held from training until migrated. CommaQA family version 3 includes the
correct evidence, case-insensitive literal search and a root schema explaining
which specialist holds each attribute. Contract migration version 3 replaces
service stores as well as NL files and discards incompatible handoff prefixes.
This is a breaking evidence-contract correction, not benchmark relabeling.

Four numeric Luna retries exhausted their request budget; execution planning
uses two provider requests per saved action reply, so 128 replies with a
256-request cap was expected. Contradictory displayed evidence was a concrete
cause to correct before increasing budgets further. Two movie queries confused
country/movie entities and movie/personal awards; their original gold is retained
and the new retrieval/schema contract is tried once. Completed service receipts
now appear after failed evals, including the return value of a successful board
write, so a rendering failure need not cause another mutation.

New conservative pending source reviews: FOLIO story 56 drops “country for life”
from the English condition in its formalization; story 8 treats publication in
1946 as writing in 1946; αNLI's Ciana dorm story does not reliably distinguish
its two proposed actions. Labels are preserved. Affected tasks are held through
collection/admission/direct materialization; no partials or held-source outputs
become training negatives. FOLIO story 24 remains a model reasoning failure,
with unchanged gold, rather than being held merely for disagreement.

Verification: Node/browser builds and browser types; 129 focused checks followed
by the added evidence-audit regression (130 distinct focused checks total), two
queue tests, the full evidence audit and migrated reference replays. New queues:
Bonsai queue-v11 (423 entries, 15 held, one reviewed board retry) and Luna queue-v7
(122 entries, two held, six reviewed CommaQA retries). Finished keys are skipped
by the shared journals. New IR/jobs are immutable and preserve source identity;
v14 rollout waits for completed case boundaries and refuses overlapping workers.
See runtime-v14.rollout.jsonl / runtime-v14.rollout-state.json for actual state.
One Luna worker/request; independent Bonsai generation. No training started.

## Bonsai container memory review

The live server still had six slots from the earlier bulk-generation setup, while
current collection allows only two simultaneous model requests. Inspection found
one model process, not six weight copies. Four slots were idle. GPU occupancy was
7,755 / 8,188 MiB; host process RSS varied around 4–5 GiB, with a 7.3 GiB recorded
high-water mark and no container swap. Docker's 8 GiB memory limit is a ceiling,
not an allocation. The host prompt cache allowed 3,072 MiB; its snapshots are
separate from GPU KV and are being reused, so shrinking it trades some recomputation
for lower RAM. The existing no-mmap loading already avoids retaining a mapped
host copy of fully offloaded weights.

Changed launcher defaults to two slots, 1,536 MiB prompt cache and a 6 GiB container
ceiling. Keep the shared 53,248-token KV buffer, q4 KV precision, checkpoint settings,
sampling, model and collector concurrency unchanged. Six-slot bulk serving remains
explicitly available. This is a resource-profile change, not a data-contract change.
Expected savings must be measured after warmup; do not describe a cold restart's
RSS drop as steady-state improvement. Reducing the Docker ceiling alone would not
release occupied memory and might kill the server under load.

`runs/bonsai-memory-20260928/before.json` preserves the old configuration and memory
metrics. The one-off restart-at-boundary.py monitor (initial PID 448361) waits for
Bonsai supervisor 428680's current case boundary, preserves partial evidence,
restarts only the exact observed container and resumes the same frozen v14 queue.
It restores the original six-slot profile if the smaller server fails startup.
Actual action/status is in restart.jsonl and state.json; after.json records the
replacement server and supervisor. Luna continues independently. Launcher syntax
and monitor compilation checked. No new generation worker or training started.
