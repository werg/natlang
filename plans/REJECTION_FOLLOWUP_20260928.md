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

## Hourly monitoring: 2026-09-29 00:13 Europe/Berlin

The requested hourly monitor is active: wait roughly one hour, inspect both workers,
review new failures independently, repair concrete problems, and repeat. No extra
workers and no training. Checkpoint artifacts: runs/hourly-check-20260929-0013/.

The smaller server restarted safely after handoff 118 finished. Bonsai supervisor
465928 resumes the same v14 queue; Luna supervisor 425858 remains independent.
After approximately 50 minutes of generation, host RSS is about 2.6 GiB and Docker
usage 2.5 GiB, versus about 4–5 GiB before. New high-water mark about 3.4 GiB, no
swap. GPU occupancy 7,157 versus 7,755 MiB (598 MiB less). This is an observed
warm-process comparison, not a controlled throughput benchmark. before.json,
after.json, restart.jsonl and warm-50min.json preserve the evidence.

All four corrected numeric CommaQA retries now finish with accepted answers
(16.4, 24.4, 19.8 and 22.2; 112, 12, 18 and 47 saved replies). The first still
uses substantial redundant work; no claim that all inefficiency is solved.
The movie-awards retry missed two titles because it matched “produced” but not
“one of the producers.” The country query still misread nationality as a movie.

**Further correction to our own schema:** writer and personal-award relations
move between the table/text specialists across worlds. A fixed ownership list in
v3 was wrong for some worlds. V4 derives ownership from the actual world KB and
uses canonical relation labels, preserving entity order, all facts, source gold,
splits and raw source caches. Audited 339,208 facts / 10,000 questions across all
splits: zero missing entities/source facts. V3 movie schemas are conservatively
held from exports until migration. Numeric v3 stays unchanged. Original surface
sentences survive in the source cache; this intentionally changes model-visible
movie evidence to make joins and complete retrieval easier.

The board retry replayed three old writes before its sole fresh teacher decision.
Its exact-once revision was already unrecoverable, so it did not test the new error
receipts. Reviewed contract migration now restarts this exact curriculum's board
handoffs from the root when their prefix contains repeated commit_move calls.
The original handoff is preserved; these restarted attempts are root retries,
not preference pairs on the old irreversible state. The exactly-once oracle is
unchanged. No global delegation restriction was added.

FOLIO 348 is now held: “Someone” becomes universal in the source formalization.
FOLIO 377 is held: “vegetation” becomes “vegetarian,” enabling a missing relation.
Other reviewed formal reasoning cases (346, 367, 395, 406) keep their labels;
poor model reasoning alone is not a reason to relabel or quarantine. Ambiguous
negation in story 378 remains under observation rather than declaring its gold
wrong. Partial/request-exhausted attempts remain checkpoints, not negatives.

Runtime v15 / contract migration v4 is prepared for case-boundary rollout. New
queues: Bonsai queue-v12 (one reviewed board retry), Luna queue-v8 (two reviewed
movie retries). Current progressing cases continue; rollout-state.json and
rollout.jsonl record actual PIDs/deployment. Verification: Node/browser builds,
browser types, 134 focused Node tests, explicit evidence audit and changed
reference replays. Course changes are committed before freezing the runtime.

Current admission confirms all four numeric retries are training-admitted (4/14
Luna final results). The longest case made five ad hoc child calls; two
short cases used named specialists only. Saved per-job result files were gathered
for this audit: the CLI summary file represents only the latest one-case invocation
and may be empty after an incomplete attempt. Original job files are intact.

Bonsai reached its boundary and now runs v15 (supervisor 495695 / queue-v12).
The root-reset board retry succeeds in 32.6 seconds / three fresh replies: one
commit, failed first render, successful render retry, exact revision 20. This
confirms the receipt guidance can be used when the teacher gets a repairable
state. Luna's v15 boundary rollout remains pending on its current progressing
case. The next hourly wait starts after this check.

## Hourly monitoring: 2026-09-29 01:25 Europe/Berlin

Both collectors reached v15 boundaries. Bonsai supervisor 495695, Luna supervisor
497979; one Luna worker/request. Bonsai server remains healthy at approximately
2.6 GiB warm Docker RAM. The canonical movie-awards retry now returns all three
correct awards and is training-admitted. The board recovery also admits. The
country query still exhausts 128 saved replies / 256 requests by repeatedly treating
Rattlider as a movie. An offline opening audit confirms the exact original question,
including “directors from Rattlider,” is visible without truncation. No missing-input
bug found. Keep this attempt incomplete/excluded; no further retries or case-specific
answer hints added during this check.

FOLIO story 416 is now held after independent countermodel review. Its existential
math/chemistry student need not be James. James's math ability can be either true
or false while his chemistry and award predicates are false; these satisfy every
premise with another student as existential witness. Both disputed math disjunctions
vary in truth under inclusive AND exclusive interpretations of “or.” Thus original
False labels are not established. The first Luna result (unknown, unknown, unknown,
entailed) was logically valid; it is a source dispute, not a teacher negative.
Six satisfying James valuations are saved in folio416-countermodels.json. Existing
static recovered/source-backed bundles contain no such story; raw curriculum shards
retain it but shared source gates prevent export until adjudicated.

**Scheduling change:** the pending queues were mostly contiguous FOLIO variants;
Bonsai spent the hour on three more 20-minute variants with little yield. Added
scripts/interleave_teacher_queue.py to rotate pending work across curriculum families
and original source groups. It preserves every eligible key, IR index, budget and
evidence path; completed attempts remain journal-skipped. New immutable IR/job paths
were prepared separately. Only held source inputs are removed (nine Bonsai entries,
two Luna entries). No dataset mix changes or target cases silently dropped. New
queues: Bonsai queue-v13, Luna queue-v9; order ledgers record every old/new position.
Folder workloads retain their expanded 672–748 request / 3,600-second budgets.

Runtime v16 adds the source hold; boundary rollout state records actual deployment
and PIDs. Verification: Node/browser builds/types, 134 focused Node checks and five
Python queue checks, including no-loss/order/budget-preservation checks. Admission
snapshot: one of three Luna v15 finals admitted (two source-held FOLIO finals), one
of one early Bonsai finals admitted. Later arriving finals are checked next hour.
Artifacts: runs/hourly-check-20260929-0125/. The hourly sleep/check loop continues.


## Hourly check 02:33, 2026-09-29 — runtime correctness before model blame

Snapshot: Bonsai 23 finals / 15 accepted; Luna 32 finals / 21 accepted on v16. Both producing fresh replies, no server or transport crash. Warm Bonsai RAM about 2.68 GiB.

Confirmed and repaired three runtime faults:
- Portable object copying and dictionary coercion lost the own `__proto__` key. Reconciliation's provided helper correctly computed 56; the runtime dropped it. Copy own properties safely and filter record fields with Object.hasOwn. Preserve ordinary output object prototypes; do not relax the eval prototype-mutation restriction.
- External service annotations such as `(proof.Step)[]` passed TypeScript but failed native parsing/resolution. Parse qualified names and resolve namespace-local aliases structurally. Invalid Step values still fail type checking.
- Saved inline NL functions captured the old eval's lexical `budget`. A subsequent eval changed it to 4000 but children still computed with 5000. Connect mutable top-level captures to the active transaction's accessors, then the committed scope. Block-local and immutable captures retain lexical behavior. Clear active accessors after success or failure.

Breaking decisions / quality gates:
- Add runtime_contract_version=17 to collection provenance and reuse matching. Old failed prototype-key reconciliation, late-binding, and affected proof-verifier runs are excluded from negative materialization, handoff-site discovery and admission. Preserve original rows; fresh retries have new keys.
- Hold KQA Pro train:33143: English asks about a nominated work; original symbolic program instead reads nomination statement-is-subject-of (award ceremony). KB separately gives Citizen Kane as for-work. No answer relabeling.
- Hold FOLIO story477: English never states TikTok is an APP or equates programmed with a program; generic APP premise has ambiguous quantification. Do not silently supply the missing connections / XOR interpretation to force gold.
- Frozen v17 was prepared, then superseded before either worker deployed it when additional source/reuse review landed. Preserve v17 and its unused queues; deploy new immutable v18. Only the exact waiting rollout monitor was stopped; no progressing workers interrupted.

V18: Bonsai queue-v15 (401 entries); Luna queue-v11 (116 entries), same journals. Reviewed fresh retries: proof chain19 root reset, Luna mutable budget and prototype-key reconciliation. Family/source rotation and large folder budgets preserved. Boundary monitor: runs/bonsai-recovery/runtime-v18.rollout-state.json (initial PID 541963).

Validation: Node/browser builds and browser type check passed; 139 focused Node cases passed. Expanded collector run had one timing-sensitive request-cap checkpoint test failure (two saved responses observed instead of three); the isolated test passed on rerun. Investigate this checkpoint timing rather than treating the expanded suite as fully passing. No training launched.

World quest failures have valid reference paths including finding the latchkey and mat; retain success certificate oracles. FOLIO171's gold Unknown is supported: heat treatment alone does not guarantee survival. Other reasoning/classification failures remain failures, not source holds.


### Same check: checkpoint race repaired; runtime v19

The flaky cap test exposed an actual slot/journal ordering race: a completed plan/action released its request slot before the driver saved its reply. A queued sibling could then hit the request cap and end execution while that completed action was still writing. Keep the pair's slot until its response is durably journaled. Replayed/planted actions use the same serialized persistence path without spending a slot. Expanded 156-test suite, including collector tests, now passes; Node/browser builds and browser types pass.

Luna's fresh mutable-budget retry on v18 accepted: first round A1/A3/A5, second round A1/A5. Prototype-key repair next. V19 is a frozen checkpoint-only follow-up using the same reviewed queues/jobs/IR and provenance contract; it does not invalidate v18 data or require another repair attempt. The exact v18 rollout monitor was replaced by v19's boundary monitor: runs/bonsai-recovery/runtime-v19.rollout-state.json. Luna already on v18 and Bonsai still on v16 at replacement; each current case finishes before its next rollout. Do not interrupt them.

Both Luna runtime repairs now accepted. Mutable-budget curriculum row is training-admitted (10 inline judgments); reconciliation retains own __proto__=56 and constructor=177, with the original urgent/unmatched/duplicate counts. The curriculum admission CLI only handles curriculum rows; reconciliation uses native materialization. Evidence: runs/hourly-check-20260929-0233/luna-v18-repair-results.jsonl and luna-v18-repair-admission.jsonl.


## Hourly check 03:50, 2026-09-29 — source audit, world affordances and invocation ownership

Both workers migrated to v19 and continue making fresh replies. Snapshot: Bonsai 12 completed finals / 7 accepted; Luna 14 / 9 accepted. Bonsai four case timeouts, Luna one timeout and one request-cap incomplete; partials show ongoing replies, not transport hangs. Repetitive policy analysis / excessive delegated command judgments are excluded as incomplete rather than mislabeled successes. Warm Bonsai RAM remains about 2.68 GiB.

- Proof chain19 root-reset retry passed and was training-admitted (opaque verifier-issued certificate). The prior qualified service type fix resolved the observed barrier.
- Audited all 166 payment judgments in payments3, isolating six gold disagreements. Hold three source messages: direct-debit refund alone does not establish an unrecognized payment; the two pending-dollar statements do not establish an extra charge. Other mismatches remain model errors. Source IDs/label-hash aliases preserved; whole affected cases held. Evidence: payment-mismatches.json and new-source-holds.json in runs/hourly-check-20260929-0350/.
- Hold FOLIO454 after checking original English and original FOL: the formal version removes animal restrictions. English permits non-animal Liam satisfying the negative branch. Two countermodels make the disputed sun/splash conclusions true or false. Preserve raw premises/formal labels; do not relabel. FOLIO425 and236 gold supported on review, retain as genuine failures.
- TextWorld v2: look() includes only currently available movement commands, without showing unvisited destinations or hidden objects. Root instructions retain exploration notes (visited rooms, observed exits, searched containers). Correct the doubled comma in iterate instructions. Contract migration v5 resets incompatible handoffs, preserves world rules/gold/source groups, and replays references before queueing. World-specific budgets now 48 turns / at least384 requests /1800 sec. Existing larger folder budgets retained.
- Luna queue extended by nine eligible pending source groups from different Bonsai families, as fresh roots; original attempts retained. One Luna worker/request remains; no workers added. One reviewed game33 retry under navigation changes.

Breaking decision: direct native exports now join action outcomes by explicit invocation_id emitted by the native agent and retained in trajectory metadata. It is not model input and does not change decoded-request digests. Older rows still use conservative opening/action matching and ambiguous rows remain excluded. Add trajectory_link_version=2 to provenance/reuse matching; use fresh job paths. A correct suppliers2 run had 19 unlinked actions under the old identical-opening heuristic; queue one fresh retry rather than inventing links. Concurrent identical openings still cannot be replayed safely for preference handoffs, so those sites/failsInPlace are held while correctly linked SFT decisions remain exportable. Preserve originals and do not guess event ownership.

Validation: Node/browser builds and browser type check pass; expanded158 Node tests pass,5 Python queue tests pass. Native export audit: seven Bonsai rows ->61 decisions; eight Luna rows ->88 decisions. One otherwise accepted Luna row was excluded for ambiguous linking and is the reviewed retry above. Curriculum admission:7/12 Bonsai and8/13 Luna (noncurriculum reconciliation uses native export).

Latest rollout: immutable runtime-v21 (496 hashes), Bonsai queue-v17 / Luna queue-v13; actual supervisor PIDs/phases at runs/bonsai-recovery/runtime-v21.rollout-state.json. Boundary monitor initially565693. V20 navigation rollout completed before v21 follow-up; no progressing cases interrupted. Audits/order ledgers/checkpoint: runs/hourly-check-20260929-0350/. No training.


## Hourly check 05:06 Berlin, 2026-09-29 (runtime v24)

- Navigation v2 retry completed game33 in 994.6 seconds/191 replies; not stuck.
  It initially placed a candybar rather than the required mat. Hold this exact trace
  (and its reused provenance) pending intermediate-decision curation. Keep the source
  task eligible; do not relabel or promote an entire episode because its goal succeeded.
- Fresh supplier retry accepted in nine turns and exports with exact invocation joins.
  This hourly snapshot's 24 accepted runs retained 488 decisions, zero unlinked actions.
- Luna's 177-message reducer finished in 1537.4 seconds/349 replies, returning 14600
  versus expected 15438. Evaluation's 0.9 agreement tolerance correctly describes an
  approximate result, but native export bypassed curriculum admission and approved it.
  **Breaking quality decision:** a shared training-quality gate now holds accepted
  partial agreement/span scores, partial file scores/failures, and pending reviews.
  Evaluation scores and raw results remain unchanged. Genuine wrong-answer negatives
  remain eligible subject to other gates. Audit of 4785 result files found 3726 accepted,
  61 held; all61 produce zero ordinary/direct/failed-run export decisions. Files may
  include repeated attempts/copies, so these are result-file counts, not unique cases.
- Current integrated static policy audit: 2192 audited references,51 rebuilt directory
  cases,72 modern recovered cases and51 source-backed cases have zero new holds.
  These counts overlap by design; do not add them as unique tasks. Historical ready
  generated exports must be rebuilt under current gates before training.
- FOLIO423 English student implications admit non-student James countermodels; source
  formalization drops guards. Hold rather than change the label. BANKING77 pound charge
  does not establish an extra/unrecognized charge; hold this specific source message.
- Repeated Boolean action shapes are expected across child reducers. Add full-request
  hash repetition alongside shape repetition and label the latter's scope explicitly.
  Neither count alone is evidence of a semantic stall. No blanket short timeout added.
- Twelve eligible unfinished Bonsai source groups transferred as fresh Luna roots;
  source IDs/splits/budgets preserved, no extra worker. Reviewed queues v18/v14 and
  immutable runtime-v24 (496 hashed files) roll at journaled case boundaries.
- Verification: Node build, browser build/types;160 focused Node tests and6 Python queue
  tests pass; diff whitespace clean. Raw data/checkpoints retained; no training started.
  Evidence: runs/hourly-check-20260929-0506/partial-quality-audit.json,
  partial-quality-export-check.json, integrated-policy-check.json and review ledgers.


## Hourly check06:30 Berlin, 2026-09-29 (runtimev25)

- Bonsai healthy PID591197/runtimev24 at snapshot,284 pending; Luna PID596968/v24,
  six pending. Bonsai container1.974GiB/6GiB, CPU/GPU generation active. Snapshot:
  nine accepted runs/seven Bonsai/two Luna →168 linked decisions, zero unlinked;
  three rejected results and four incomplete/timeouts. Later Luna supplier672 exhausted
  its128-action/256-request budget and rolled to v25 at the journaled boundary.
- FOLIO409: English P1 is (!Benefit && !Apple) →(Fruit && Red). Source FOL instead
  uses !(Benefit && Apple) →RedFruit, a different antecedent, and drops fruit guards
  in other rules. Enumerated12 models of the English singleton rules; three countermodels
  to C2's asserted entailment and four to C3. Hold this source story; do not change gold.
- EntailmentBank LEAP__7_10338: source proof infers instinctive behavior from being
  inherited, and adaptation from positive survival impact. Both reverse implications;
  the store does not establish the missing directions. Hold source across shape/group
  aliases, generation/admission/negative exports. Do not punish a model's justified refusal.
- Two large payment reducers exhausted672-request budgets at336 saved replies.
  One set200000ms on a158-message child batch, causing a timeout while child work
  could continue; another used untyped children and incorrect argument names, then
  repeated classification. A short eval timeout includes queue/model wait and does not
  cancel started work. Tool schema now explains this and advises omission for large
  batches. No default wall timeout reduced, no blind budget increase/retry of bad sources.
- Two further pending-dollar texts have no evidence of additional/unrecognized charges;
  exact canonical IDs/legacy aliases added to pending source review. Filtered73 Bonsai
  queue entries and7 Luna entries (counts include completed entries). Preserve raw cases.
- Twelve unfinished source groups without an accepted Luna result and without a pending
  Luna attempt added as fresh roots. Some are deliberate retries of earlier model failures,
  not new unique source tasks. IDs/groups/splits preserved, max12 transfers this check;
  one worker. New queuesv19/v15, runtimev25 frozen with500 file hashes. Monitor
  runs/bonsai-recovery/runtime-v25.rollout-state.json determines actual processes.
- Verification: Node/browser builds and browser types;161 focused Node checks pass.
  Integrated policy audit still holds zero of2192 references/51 directory/72 modern/51
  source-backed cases (overlapping counts). No training. Evidence in
  runs/hourly-check-20260929-0630/ includes snapshots, countermodels/raw source proof,
  source holds, queue orders and migration ledgers. Continue hourly loop.
