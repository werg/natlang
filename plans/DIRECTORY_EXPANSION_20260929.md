# Reducer/source expansion and Bonsai concurrency — 2026-09-29

## Decisions and course changes

- User target: directory and file reducers supply at least **25% of unique admitted train decisions** in the next complete mix. Tree edits are separately counted. Meet the target with more verified supply; no duplicated samples or silent removal of other families. The generated pipeline now checks the final audited joint corpus before training. This is a deliberate new blocking gate; a corpus below 25% needs further generation.
- Expand beyond files: TreeDST supplies edits to existing dialogue goal trees with complete prior conversation context and system acts. Only bounded explicit-value transitions qualify; implicit, broad, operator/domain changes and unsupported reorderings are held.
- Add QASPER paper extraction and SciFact abstract/claim assessment to existing CommitPackFT, TAT-QA and MuSiQue expansion. Original train splits, source groups, licenses and content hashes are preserved. Development overlaps are excluded where source IDs establish them. SciFact test claims expose no cited-document IDs; no test labels are imported.
- Acquisition uses a separate cache (`vendor/directory-sources-expansion-20260929`), preserving the original pilot. TreeDST data is git-pinned. QASPER/SciFact data archives and rows are content-hashed; their revision fields identify pinned acquisition metadata, not a claim that the data archive came from that git commit.
- Initial broad builds exhausted Node heap by accumulating all replays. Builders, bundle validation and materialization now stream one verified case at a time, bind checksums, and publish manifests last. Repeatable `--exclude-manifest` removes already integrated logical source cases.
- Initial teacher assignment based on global modulo skewed source families toward Luna. The broader additions use per-source counters and mixed queues; one Luna worker remains sufficient. Original attempts remain in place.
- TreeDST's author implementation keys children by name. Added a strict named-tree comparison that ignores sibling order but rejects duplicate names, missing/extra fields, malformed nodes and changed values. Older failed cases under the array-exact oracle are held from training negatives; originals are not relabeled.
- TAT-QA had a correct answer rejected for JSON whitespace. Added JSON string-record comparison for answer and output files: formatting/key order are immaterial, values/schema/source preservation remain exact, duplicate keys are rejected. Older failures under the formatting oracle are held from negatives.
- A fresh TAT-QA retry exposed a signed-expense ambiguity for source UUID `f944b361-6e00-45c8-a7e1-1f5c6e0fd6b1`: table entries(285)/(267), gold average276, equally direct signed average-276. Hold the source from positives and negatives until its convention is independently established. No gold rewrite. Source review now recognizes the `source` field on modern adapters as well as legacy `dataset`. V6 refresh removes this case; preserve V5 as historical evidence.
- QASPER gold span boundaries can reject semantically equivalent answers. Wrong-answer traces are preserved for independent review and excluded from negative training until equivalence is established. Source gold stays unchanged; default scripted answer/reasoning decisions remain held.
- User approved roughly four or five Bonsai requests. Set **four server slots and one global four-request semaphore including children**; queue batches four independent roots in one collector. Provider queues remain single-root, default cap one. Supported local batches are one through five, but active profile is four.
- Retain 53,248 shared q4 KV tokens, collector weighted budget 40,000, no mmap, one model copy, host cache 1,536 MiB and 6 GiB container cap. Four-slot defaults retain this smaller memory profile. Switches occur only after journaled batch boundaries; exact supervisors are inspected and overlapping collectors refused.
- Fixed multi-root activity accounting: a completed sibling's result counts even while another sibling has a partial checkpoint. Repeated action shapes remain distinct from repeated full request hashes.

## Evidence and operational state

Audit directory: `runs/directory-expansion-20260929/`.

- Expanded bundle: 1,604 additional native cases after excluding 51 pilot identities: CommitPack287, TAT-QA255, MuSiQue288, QASPER300, SciFact174, TreeDST300.
- 4,113 approved native decisions, 1,049 upstream held decisions. Tree edits contribute600 approved decisions; directory cases contribute3,513. No model calls in static conversion; synthetic reasoning notes are context only.
- V6 final validation/render/token audit passes all4,113 at8,192 tokens. The combined static preview passes all9,108 decisions with zero duplicates or holdout overlap:3,670 directory reducer decisions (40.29%),600 tree edits,4,838 other. The earlier V5 preview9,114 is preserved; six decisions were removed for the signed-expense source hold. Final complete teacher/student-tokenizer mix must be re-audited. Proofs: `verify-v6.log`, `verified-v6/{joint.ready.jsonl.audit.json,joint.mix.json}`.
- Published corrected bundle: `data/teacher/directory-expansion/static.manifest.json`; earlier publications preserved in `runs/directory-expansion-20260929/{previous-published-bundle,published-v5-preserved}/`. Both default training entry points discover it. The original4,995-decision integrated static preview remains preserved.
- Full SWE issue trajectories converted: **zero** in this expansion. Keep unsupported environments/dependencies and unknown success held. The earlier eight independently specified creation tasks remain in the original pilot; do not claim they are full issue trajectories.
- Four-slot rollout: `four-slot.state.json`, append-only `four-slot.rollout.jsonl`, queue-v24. 1,229 pending roots compacted into308 batches, including307 full batches. Frozen runtime-v29. No partial root with saved progress was interrupted.
- First measurement:272.75sec, mean3.68 active slots, max4,18.27 generated tokens/wall sec, GPU7,455–7,457MiB, server RAM peaked~2.85GiB. Includes cold prompt processing and incomplete request counters; it is not a matched throughput comparison with the historical two-slot average. Keep four until longer usable-data throughput evidence supports five.
- Runtime-v30 adds JSON scoring and QASPER negative holds; both migrations completed at boundaries to Bonsai queue-v25 / Luna queue-v20 (then queue-v21 for two reviewed JSON-format retries). Actual Bonsai PID/state in `v30.rollout-state.json`, latest Luna in `json-repair.rollout-state.json`; old/new queue and raw artifacts stay intact. Luna retains cap1.
- Final focused checks:66 Node source/oracle/curriculum/adapter tests,29 source-review/oracle checks and22 Python queue/combine/static/mix/rollout tests; build/browser types passed. Added a materializer CLI regression and fixed its aggregate summary (the streaming rewrite left a loop-local variable in final logging). Four broader-source and14 earlier oracle checks also passed. Existing training-recipe fixture failures are not claimed fixed.
- No training or student model weights loaded. Token audits use CPU tokenizer-only containers.

## Follow-through

V6 validation and publication are complete; frozen runtime-v31 validates the published bundle. Both v30 migrations completed. Two independently verified whitespace-only TAT-QA failures received fresh Luna attempts under corrected scoring; historical outcomes stay untouched. Published bundle also validates with the frozen `pipeline-runtime/`, which contains the corrected streaming materializer CLI summary. Generation snapshot:64 results,37 admitted,177 linked approved decisions, one held decision, zero unlinked exports (`generation-quality-review.json`). Audit new four-root results using current gates. Continue hourly generation checks and rejection investigations. Runtime-v31 adds the source hold and corrected materializer summary. Both migrations completed at boundaries: Luna queue-v22/PID695902, Bonsai queue-v26/PID701072. Verify recorded state and exact live command before acting. Actual state: `v31.rollout-state.json`. Local supervisor stall detection now checks real server decode progress as well as saved replies, preserving the300-second no-progress timeout and hard batch budget. This prevents a slow, actively decoding batched reply from being called stuck. No extra model requests or cache capacity.

The final complete training pipeline recomputes the25% requirement; a static preview meeting it does not establish the eventual joint mix.

Future candidates: SMCalFlow revise/refer graph edits, SParC SQL AST revisions, and WikiIns instruction edits. They need license/split/independent-oracle review before import; no compatible execution trajectories claimed yet.

## Luna operational retry policy (2026-09-29, user requested exponential backoff)

- Existing collector backoff capped before jitter and did not cover provider HTTP500/usage limits or record waits. Supervisor immediately advanced on failed execution.
- Centralized transient classification and bounded exponential jitter: provider transport base15s/cap30s; rate base45s/cap120s. Retry count unchanged (one transport retry, up to three rate retries in queue runs). Zero delay remains available for fixtures.
- Respect supplied Retry-After headers (seconds/dates), retry_after_ms/retry_after/resets_in_seconds and explicit retry-again seconds exposed by SDK errors. SDK can flatten/drop headers: unavailable values cannot be recovered. Provider minimum takes precedence over policy caps. Hard case wall budget still applies.
- Atomic job retry deadlines plus terminal error deadlines; logs contain attempt/reason/wait/deadline. Abortable waits clean up metadata. Deliberate provider wait is activity, not a stall. Timeouts preserve minimum delay before subsequent requests.
- Supervisor failed-run cooldown doubles30/60/120/240 to300s cap with jitter. Finish journal records failure streak and next_allowed_at; restarts observe remaining delay. Completed executions reset streak, including ordinary rejected answers. No operational waits become training turns.
- No concurrency change: one Luna/request, four Bonsai. Boundary rollouts v32 then v33; v33 finishes long minimum-delay persistence. Check actual state in v33.rollout-state.json. Runtime built from immutable v31 with only collector/retry source+compiled overlay, to exclude unrelated concurrent adaptation edits.
- Build succeeds;21 Node checks against frozen v33,13 Python checks pass. Covered doubling/caps/jitter/headers, malformed delays, cancellation with no result, cooldown/reset/restart and durable exhausted-retry deadline.
- No source gold or rejection criteria changed by retry policy. No training started.

- v33 boundary migration complete: Luna PID731206, unchanged queue-v22/cap1. First v33 case completed in34.7s, then advanced normally. Bonsai PID701072 continues immutable v31/cap4. Latest expansion audit156 results/91 admitted/391 decisions/9 held/0 unlinked. Commit2aea07a implements policy.

## Rejection audit 09:26–09:38 UTC (2026-09-29)

Audit: `runs/generation-check-20260929-0926/`.

- Live check: four-request Bonsai PID701072/v31; single Luna PID731206/v33. GPU snapshot98%,7457MiB. Since v33 deployment Luna completed53 runs with zero operational failures/cooldowns. Bonsai had two completed batches and one hard wall timeout (3000s plus10s shutdown), with97 fresh replies and49665 completion tokens, continuing decode activity. Three roots left partial checkpoints; one finished result survives. This was a slow budget exhaustion, not a frozen GPU. Originals remain; no unreviewed automatic retry.
- Fresh expansion review:65 results beyond prior156,37 admitted before new holds;12 QASPER answer mismatches,11 TreeDST tree mismatches,2 SciFact issues,1 escaped CommitPack edit,2 constructed/source logic results. QASPER mismatches include valid citation omission/answer-span boundaries; retain existing negative holds. Do not rewrite gold or declare every rejected answer incorrect.
- Source holds: SciFact claim466 cites abstract22544171, which lacks the progerinonly/Lmna knock-in evidence. Claim576 cites abstract4468861, which assigns exhaustion reversal to addition of PD-L1 blockade, while its annotated sentence does not establish the specified anti-CTLA4 effect. FOLIO story24 lacks Leetcode-problem membership for named objects despite guarded English rules. Preserve all original labels; hold positives and negatives, including historical repairs/source groups.
- Breaking quality decision: current TreeDST wrapper does not supply an independent branch ontology/transition contract. Substring-present leaves do not establish where to put new branches or which old branches to remove. Quarantine such tasks with `unverified_tree_transition_contract`, including historical accepted/static cases. Only literal replacements in existing single-value equals/notEquals slots qualify; branch deletion/addition, operator changes, duplicate children and unknown comparator layouts do not. The same check is enforced by adapter and training admission. Retain21/300 static TreeDST cases; hold279 pending independent ontology/schema work. Do not give agents gold-derived layouts.
- Prompt correction: supplied exact structured text should be parsed and used directly instead of escaped/retyped in code, with file verification. One fresh Luna CommitPack repair in separate `v34.payload-repair.jobs` completed accepted in four actions: read request, readJson/editText using fields, diff, return. Old wrong-backslash trace preserved. This is a useful repair, not proof all future edit errors are solved.
- Queue-v27 Bonsai and queue-v23 Luna remove204/95 source/contract-held roots respectively (counts include historical attempted entries). Existing batch/member attempt identities, source files, seeds, checkpoints and wall budgets preserved; contiguous valid groups retain shared collectors/cap4. At preparation1004 Bonsai and279 Luna valid pending roots. Luna queue-v24 prepends the one reviewed edit repair, same single worker/cap1.
- Frozen v34 uses immutable v33 plus policy/source-review/tree-contract/prompt files and broader adapter only, excluding concurrent adaptation/evaluation edits.52 frozen Node checks,25 Python queue/rollout/static/mix/combine checks and build pass. Earlier integrated source-backed/recovered static manifests validate under v34.
- Luna migrated at a case boundary to v34, then queue-v24/PID1100034 for the repair. Bonsai migration remains pending at its progressing batch boundary. Actual state: `v34.rollout-state.json` and `payload-repair.rollout-state.json`; verify before action. No GPU server restart or training.
- Stricter teacher audit at09:37:240 results/128 admitted/575 approved decisions/17 held decisions/zero unlinked. Whole-task holds are excluded, so these decision counts are not additive with earlier previews.
- Static V7 rebuilt without model calls:1323 cases,3553 approved decisions,1047 upstream held. Sources: CommitPack287,TAT-QA255,MuSiQue288,QASPER300,SciFact172,TreeDST21. Token/joint verification and publication tracked in `verify-v7.log`; preserved V6 remains historical.

- V7 verification and publication completed: all3553 expansion decisions and8548 combined static decisions pass the8192-token audit, zero duplicate pairs and zero holdout overlap. Final static mix3668 directory reducers(42.91%),42 tree edits,4838 other. Published `data/teacher/directory-expansion/static.manifest.json` validates under frozen v34; all three integrated manifests validate. V6 original files preserved at `runs/generation-check-20260929-0926/published-v6-preserved/`. The final complete mix must still be re-audited after generated trajectories are added; static preview is not the complete target mix.

## Quality check and Luna semantic sweep — 2026-09-29, 10:55 UTC onwards

- Initial expansion snapshot:371 results/203 admitted/889 approved decisions/110 held decisions/zero unlinked. Since prior check Luna completed105 runs, Bonsai7 batches, neither had operational failures. Latest v37 admission snapshot:411 results/237 admitted/1,025 approved decisions/117 held decisions/zero unlinked. These are evolving teacher results, separate from verified static decisions.
- **Course change: pause QASPER live generation** until an extractive-equivalence oracle is available. Fresh QASPER45/49 rejected and only4 admitted: wrong-answer cases often conflate span/format mismatch with reasoning failure. Static document-read decisions and existing valid matching results remain eligible; reference collection remains available. No gold rewritten. New hold reason is `awaiting_extractive_equivalence_oracle`.
- Independently confirmed new source holds SciFact657/748 (wrong protease/causal mechanism), FOLIO406 (explicit satisfying countermodel contradicts claimed C3). See `runs/generation-check-20260929-1055/folio406-countermodel.json`.
- User-authorized `gpt-6-luna` subagent completed a read-only semantic audit of all170 remaining SciFact cases in immutable bundle-v8. Initial144 pass/26 review was corrected to150 pass/20 review after root noticed omitted second abstracts. Both original and corrected audit are preserved. **Do not treat a subagent verdict as a source label**: inspect every supplied document, distinguish unsupported from contradicted, and compare with annotated labels only after assessment. Claims542/563 are valid CONTRADICT negatives;71 is not a decisive contradiction because damaged but surviving cells can still be viable. Claim164 is held for missing subtype evidence, not relabeled as contradicted.
- Root reviewed every original flag against all supplied abstracts and also checked the bariatric-cancer group. Fifteen additional cases are held pending adjudication: claim:79, claim:104, claim:160, claim:164, claim:165, claim:168, claim:169, claim:200, claim:258, claim:331, claim:372, claim:397, claim:564, claim:584, claim:747. Details/document IDs and all170 coverage proof: `runs/generation-check-20260929-1055/luna-quality-sweep/root-adjudication.json`. Other flags retained when second abstracts or ordinary supported scientific inference resolve them. Includes keeping KLF4 cases668/669, charcoal245/246, association709, antimicrobial121/122. Source gold, IDs, revisions, groups and licenses stay intact.
- Prompt improvements: span find/replace must preserve surrounding text including final newlines; iteration `until` checks initial state, so initialize pagination `more:true`, read/accumulate every page and advance page numbers. The latter also appears in the depth-limit prompt, while new ad hoc NL delegation remains disabled there. Frozen v36 caught a missing depth-limit paragraph in testing and was never deployed. V37 corrected it;20 focused frozen Node checks pass,25 queue/rollout/static/mix Python checks passed earlier in this check. Selective overlays exclude concurrent adaptation work.
- V8 was published after strict native validation/rendering/8,192-token audit/deduplication/holdout/mix checks:1,321 expansion cases/3,551 approved decisions/1,045 held;combined8,546 static decisions,3,666 reducers(42.90%),42 tree-edit. V7 preserved. Root's15 further holds produce V9 candidate1,306 cases/3,536 approved/1,030 held; final publication verification underway, see `verify-v9.log`.
- Queues filtered before rollout: Bonsai v28→v29 removed10 source-held roots;Luna v25→v26 removed5. A separately logged fresh newline repair was prepended to Luna queue-v26; original rejected trajectory retained. QASPER generation pause had earlier removed197 Bonsai/93 Luna roots. Split batches preserve original member attempt identities and do not duplicate attempted roots.
- The pending v35 boundary monitor(PID2106976) was superseded by exact monitor-only SIGTERM while Bonsai was waiting and Luna already running. No worker was interrupted. V37 boundary monitorPID2112978; Luna migrated at its completed boundary toPID2113145, queue-v26/cap1. BonsaiPID1130311 stays on runtime-v34/queue-v27 while its batch continues decoding; planned v37/queue-v29/cap4. Inspect `v37.rollout-state.json` for actual final state. Latest GPU97%,7,457MiB; Bonsai batch51 fresh replies/28,806 generated tokens and increasing, not a silent stall. No student training or weights changed.

### Final quality-sweep publication

- V9 finished all validation/rendering/8,192-token/deduplication/holdout/mix gates and is now published:1,306 expansion cases/3,536 approved decisions/1,030 held;combined8,531 admitted decisions,3,651 directory reducers(42.7969%),42 tree edits. Zero duplicate rendered pairs or train/holdout overlap. Published-manifest validation passes, as do original source-backed/recovered bundles under v37.
- **Publication shape improvement:** manifest now references immutable `v9.*` payload names, installed before atomic manifest replacement. Old payloads remain valid for readers holding the old manifest. Use `static.manifest.json` as authority; legacy unversioned payloads are historical. V8 also preserved in `published-v8-preserved/`. Rejection/trajectory ledger names are recorded in `quality_audit` metadata.
- Final21 focused frozen Node checks pass, including retention of valid CONTRADICT negatives and second-document evidence. Fresh newline repair admitted successfully. Final dynamic snapshot421 results/246 admitted/1,070 approved decisions/117 held decisions/zero unlinked. Luna continues on v37; Bonsai remains actively decoding under v34 until its boundary, with exactly one v37 rollout monitor. No training.

## User-requested shutdown — 2026-09-29 11:28 UTC

Stopped exact migration monitor first, then both supervisors via SIGTERM/collector cancellation;
closed Bonsai container and verified all generation processes gone. No completed result or
checkpoint deleted. Recorded intentional shutdown events (no finish/failure) and checksummed5
active-case artifacts. Durable checkpoint:`/home/werg/natlang/runs/shutdown-20260929-112702/checkpoint.json` (pointer:`runs/SHUTDOWN_LATEST`).
Bonsai interrupted52/55 retain45/47 turns;53 rejected and54 accepted results preserved. Luna537
retains3 turns. Pending next queues364 Bonsai entries/98 Luna entries. Resume old Bonsai batch
alone under v34 before v37/queue-v29; Luna resumes v37/queue-v26 directly, existing journals.
Do not reuse historical PIDs or old rollout config. Full exact server/worker commands and
quality priorities are in HANDOVER.md. Computer shutdown is handled by user; no auto restart,
no training. Unrelated working-tree changes preserved.

## Generation resumed after reboot — 2026-09-29 16:51 UTC

Verified frozen v34/v37 runtime identities, installed dependencies, all5 saved partial/result
SHA256 hashes and V9 published payload checksums. No previous generation/server process was
running. Restarted natlang-bonsai detached with same4 slots/globalcap4,6GiB memory cap,
1,536MiB host cache; health endpoint OK, model loaded, observed GPU100%/7,455MiB/59.72W.
LunaPID5040 resumes v37/queue-v26 and existing v2 journal; interrupted537 completed accepted
in20.3s(4 turns), next540 started. BonsaiPID5100 resumes only interrupted v34 batch;collector
reported2 already-complete roots and2 pending. New exact-PID boundary monitor5101 moves
Bonsai to v37/queue-v29 after that batch, preserving journal/member identities and no overlap.
Actual rollout state/metadata/logs: `/home/werg/natlang/runs/restart-20260929-165057`; pointer:`runs/RESTART_LATEST`.
Shutdown history remains preserved. Prior quality holds, QASPER generation pause and staticV9
remain unchanged. One Luna worker/cap1, no extra repair workers, no student training.

## Post-reboot progress, failures and course changes — 2026-09-29T19:10:50.692329+00:00

- Bonsai migrated exactly at its recovered batch boundary to v37/queue-v29,PID5386;
  one supervisor, cap4, GPU100%/7,457MiB. At the check,356 queue entries remained.
  Since reboot six supervisor batches completed and two reached hard timeouts.
  Last timeout `handoff:686` had171 saved turns/130 fresh replies at the wall,
  and GPU decode was increasing, so it was genuinely slow rather than a stuck
  provider. Its completed member results remain; unfinished FOLIO425 has63 saved
  turns, but a separate FOLIO425 result is already accepted. Avoid blind reruns.
- Luna v37 queue-v26 finished98/98 remaining attempts, all supervisor status
  `complete`, no provider failure. It then exited normally. Latest teacher audit
  546 total results/361 admitted/1,629 approved decisions/164 held/zero unlinked.
  Delta from prior audit:125 results/115 admitted/559 approved/zero unlinked.
- Reviewed10 fresh model rejections: five MuSiQue, four native inline, one
  CommitPack. The CommitPack oracle accepted while an intermediate scope error
  was recorded; admission still held it. MuSiQue `3hop1__437119_23998_21435`
  answered with the correct long gold phrase plus subject “Yale” and was
  rejected by exact normalized wording. The other MuSiQue answers missed the
  target or a document link; `4hop3__672860_75897_8509_19700` does have a
  supplied German-migration/Brazil five-million evidence chain. No source gold
  changed and no invalid negative was admitted.
- **Compiler bug:** final `typeof v, v` was emitted as `__natlang_finish(typeof v, v)`.
  The second argument is bindings; a string value became numbered locals `0`–`4`
  via Object.entries, after which every eval failed with invalid injected binding.
  Parenthesize final expressions and explicit returns when lowering so comma
  expressions remain one argument. Frozen v38 contains this fix. Also added a
  fail-closed runtime guard rejecting invalid eval binding names before state
  commit; frozen v39 contains both changes. Builds passed; frozen identities
  verified. No unit tests were added or run during this status check.
- The first v38 boundary monitor rejected an absolute old-queue spelling that
  did not match the live process command; it stopped without touching workers.
  A corrected v38 monitor waited safely, then was intentionally superseded by
  v39 monitorPID8804 while still waiting; no worker was interrupted. v39 uses
  queue-v31, a copy of already filtered v30 with one fresh, separately logged
  repair of the poisoned inline case prepended. Current v37 Bonsai case
  `handoff:658` is decoding and checkpoints are increasing; v39 deploys only
  at its journaled completion. Actual monitor state:`runs/check-20260929-190056/v39.rollout-state.json`.
- One Luna workerPID8658 restarted on frozen v38/queue-v27 with five reviewed
  MuSiQue retry attempts (four previous Luna misses, one previous Bonsai miss),
  preserving original results and gold. One worker/cap1; stop after queue ends.
- Future MuSiQue source conversion now uses QA span-F1 threshold0.9. This admits
  a close long answer with one named subject but still rejects wrong short
  dates, numbers, and directions. This is a forward adapter contract change;
  existing prebuilt IR/static V9 remain unchanged. Rebuild/validate new source
  slices before attributing any gains to this new oracle.
- Quality priorities: inspect the poisoned-case fresh repair under v39; evaluate
  retry outcomes before more attempts; continue held-source/QASPER filters;
  monitor reducer share. No student training or weights changed.
