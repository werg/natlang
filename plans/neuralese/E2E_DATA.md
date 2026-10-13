# End-to-end Neuralese data: producer→consumer pairs at scale (2026-10-12)

The e2e stage (recipes neuralese-e2e-v1 / -mellum-v1 / -lfm-v1) trains on pairs in which one natlang call's result is
read by a later call. The current corpus, neuralese-e2e-union-cohort-20261012-v1, has 1,389 records and 1,137 edges
(769 train records). Target: 50k+ authenticated edges across many domains, tokenizer-agnostic records (one file serves
both lines in lockstep).

## What counts as a pair (unchanged)

An edge is a `gradient_edge` reference of `train/trajectories.authenticated_recurrence_context_view`: a `read` part
whose name is a producer record's target `$write` and whose body equals that write's source. Since conversion /15
(`ts-host/src/compiler/neuralese-conversion.ts`) a producer is a **model-written** value:

- `result:` child results: a child call's direct `return_result` value (>= 16 chars), printed by its caller;
- `argument-read`: a structured child result's text field passed into another call;
- `inline-site:` a caller's `nl\`...\`` literal instruction, read by each child call it runs;
- typed `Neuralese<string>` results and `soft-state:` writes.

Not producers: a value the child computed in eval and returned (`observed-host-result`, a host capture), a short
crisp value (`crisp-value`), crisp tool output. This is the rule that decides every yield below.

## Routes measured today

| Route | Measured | Edges | Value diversity | Cost |
|---|---|---|---|---|
| (a1) Call store (snapshot `natlang-corpora/callstore-snapshot-20261012`) | 936k calls; 443k child outputs >= 16 chars read by a parent, 14.4k sibling edges, all from reference/scripted executors; real-model producers (Qwen3.6 teacher, pi/apps): 1,434 child→parent + 156 sibling | ~1.6k usable (real-model part) | high | export work only |
| (a2) Static replay of every curriculum family (seed 41001, 4 shapes, `--authored-plans --supervise-reference-answers`) | 83 families; edges only from `inline-site:` writes; 0 child-result edges (children compute in eval → host results; or crisp labels) | ~3.9k per 4 shapes, unbounded by seed | ~70 distinct written values in all (authored constant instructions); folder families reach 190 readers per program of one instruction | zero |
| (b) New data-backed hop-chain family (MuSiQue 19,938 train + 2,417 dev; HotpotQA) | in construction (children return the gold supporting sentence directly; later hops read earlier facts) | est. 3–7 per question: ~70k from MuSiQue, more from HotpotQA | one distinct value per hop | zero |
| (c) Harness / SWE trajectories (489 + 31,529 trajectories) | no `$write`, no `read`; tools are crisp; no sub-agent calls | 0 | — | — |
| Registry fold-in (every registered corpus with read parts, same rule as v1) | scan running | expected low thousands at most: most registered recurrence files are byte or row copies of the 4,961-row assembly | mixed | CPU only |
| Luna real-model collection (Pop) | v28: 21 edges / 5 programs, v29: 85 / 5 (conversion /15) | ~4–17 per program | high | Luna provider, 5–15 min per program per worker |
| Teacher :8082 re-execution of union IR (s72 2,772 cases, campaign, bgkit) | teacher not running (GPU held by Mellum v4) | by family, as Luna | high | teacher-hours when the GPU frees |
| Clef decisions | Clef only picks among oracle candidates (accepted iff it agrees with the oracle); the written values are the authored reference, so it adds no producer the static route lacks | as (a2) | as (a2) | ~400 input tokens per program |

### The call store, in detail

- Snapshot: sqlite `VACUUM INTO` (2.4 GB) plus hardlinks of all 2.75M content-addressed blobs (immutable files named by
  hash; the store deletes by unlinking, so the links keep them at no space cost).
- Eviction (`ts-host/src/calls/store-core.ts` `evict()`, run every 200 writes): above `maxStoreBytes` (50 GiB default)
  or below `minFreeBytes` (20 GiB) free, it drops unpinned event streams oldest first, then whole unpinned calls, down
  to 90% of the bound. The live store was at 49 GiB.
- 767k of the 936k calls have no model id and ~4 tokens out per call: curriculum-reference replays and app test
  drivers. Their child values are authored or host-computed, so they are not model writers under the rule above.
- The store holds call records and reduction traces, not native turn records. The same reference programs are
  regenerated deterministically through `build-static-lambda-corpus.mjs` (route a2), which emits native records.
- The real-model part (~1.6k edges, wiki-maintain / wiki-merge / pi / compilers stages) needs an exporter from the
  call store to native records (collector rows → `materializeNativeRows`). It is queued behind route (b).

## Quality checks (every route)

- Authenticated edges only (`authenticated_recurrence_context_view`), the producer approved, the body exact.
- Splits: the rows' own split; cross-split edges dropped; train edges whose endpoints share a source group with any
  held document of union r3 (all renderings) dropped, as in v1. The hop-chain family uses MuSiQue's seed groups
  (`musique:seed:<id>`, `directory-sources.mjs`) and MuSiQue dev as its held split.
- Diversity: report distinct written bodies beside edges; cap readers per distinct body at 64 for the static
  inline-site supplement, so constant instructions cannot dominate.
- Dedupe across sources by record id (first source wins) and by piece name (a conflicting piece holds the later
  record), as in the v1 builder; edges keyed by (writer, reader, name, body sha256).
- Exact oracle at the root where the task has one (MuSiQue answer + aliases), for the e2e task-success gate.

## Renderings

The source rendering (full: system soft pieces and the runtime tool schemas) is the training contract today; Pop runs
the e2e arm with a recipe-declared full rendering. A program-header rendering of the same records is emitted beside it
by `scripts/neuralese_data/render_e2e_program_header.py` (shared `data/e2e_rendering.program_header_native_record`),
which requires the re-authenticated edges to equal the source edges. The two renderings are compared on the same e2e
qualification and the better one is adopted for both lines. v1.1 = program-header rendering of v1.

## Schedule

| When | What | Edges (cumulative, est.) |
|---|---|---|
| 2026-10-12 | v1 (1,137); registry fold-in; v1.1 program-header | ~1.1–2k |
| 2026-10-13 | hop-chain family: MuSiQue train+dev through the static pipeline (CPU, ledger, low priority); capped static inline-site supplement; e2e corpus v2 registered and bound in neuralese-e2e-v1 | ~50–75k |
| 2026-10-14 | HotpotQA hop chains; call-store real-model exporter | ~100k+ |
| ongoing | Pop Luna multi-call programs (Neuralese<string> child results, conversion /15), same record format; :8082 teacher re-execution of union IR when the GPU frees | +4–17 per program |

Expected date for 50k authenticated pairs: 2026-10-13, carried by the hop-chain family (zero teacher cost). Model-
written diversity beyond gold quotes comes from Luna and the teacher.

## Pop's share

Pop's six Luna workers produce the same record format (`natlang.teacher_training_turn.native/1` through
`materialize-native-teacher.mjs` and `neuralese-convert-trajectories.mjs`, conversion /15): multi-call programs whose
child calls return `Neuralese<string>` or long text values that a later call reads. Proposed share: two of six workers
on e2e programs (about 4–17 edges per program, 5–15 minutes per program per worker: ~2–6k edges/day), with the v29
source-world families as the starting set. Records are tokenizer-agnostic, so both lines use the same files.

## Status 2026-10-13: e2e corpus v2

Owner 2026-10-12: include everything (only exclusions: held splits, exact duplicates, records that fail conversion,
each registered with its reason). Built with `scripts/neuralese_data/e2e_mine.py` (shard by trajectory → convert
with `neuralese-convert-trajectories.mjs --admit-host-results`, conversion /16 → authenticate each trajectory with
`authenticated_recurrence_context_view` → assemble whole chains, dedupe, split rules, omissions.jsonl).

- **neuralese-e2e-cohort-20261012-v2** (registered; HDD, /mnt/external/natlang-development-data/e2e-data/corpora):
  95,106 edges (85,315 train / 9,791 held), 24,141 chains, 174,608 records, 80 families. Largest: musique_hops 71,103;
  edit_stream 4,852; repo_answer 2,619; digest_desk 2,244; translation_desk 1,969; knowledge_evidence 1,450.
  Omitted with reasons: 3,288 duplicate ids, 243 piece-version conflicts, 441 conversion failures, 34,773
  authentication exclusions (mostly whole shards of campaign-uncovered/s72 that failed on a duplicate writer name;
  fixed by per-trajectory authentication and being redone).
- **neuralese-e2e-cohort-20261012-v2-slice1** (registered; NVMe): the trainer-sized slice, 7,007 records (6,003
  train / 1,004 held), 7,521 edges, 80 families; bound in neuralese-e2e-v1 for both lines (1e8c1df9).
- **musique_hops** (e284901a) carried the bulk at zero teacher cost: 75,602 records from 17,424 MuSiQue questions.
  Conversion /16 also turned on child-result edges in every family whose children return long text directly.

### The call store, final finding

An exact export of call-store runs to native turns is not possible: the reduction trace stores message counts, not
the messages (opening, rendered tool results, reasoning, offered tools), so the materializer's per-decision context
cannot be rebuilt. Its edges come from: port-record replays (372k child→parent calls; the same runs exist as
registered native turns, port-record-calls-s3 v1/v3 and view-ask v3, now being mined), curriculum-reference replays
(341k; regenerated by the static campaign, mined), scripted app/test drivers (~30k, stub answers, not training data)
and ~3k real-model calls (pi app, specializer; not exactly rebuildable). Store bug for its owner: reference runs share
a runId (`curriculum-reference`, `native-run`), so `INSERT OR REPLACE` keeps only the last run's root call.

### In progress (v3)

campaign-uncovered and s72 redone per trajectory; static-curriculum-s7k b00 (140 shards, oversized shards split by
trajectory) and seeds s7011–s7014; v13-r2 (8.4 GB); port-record-calls s3 v1 (~140 GB decompressed), s3 v3, view-ask
v3. Real-model breadth: Pop's two Luna slots on the shortlist (musique_hops seed 52001, teacher-only families,
long-text return families, v29 worlds).

### Supervisor audit, 2026-10-13 22:04 UTC

The five sources previously described as never mined have completed shard conversions and are already included in
the immutable `neuralese-e2e-cohort-20261013-v3` receipt: v13-r2 (54 shards, 4,343 pre-assembly edges),
campaign-uncovered (20, 3,133), s72 (28, 2,001), static-lambda-v2 (15, 7,539), and bgkit-knowledge (32, 6,871).
These are conversion counts before assembly deduplication and held-group exclusion, not additional admitted edges.
None of those sources has an unfinished or failed shard in the current work directory.

Mining execution now uses the canonical checkout for both the Python miner and the TS converter. The retired-worktree
unit was stopped by exact name and replaced by `e2edata-mine-canonical`, low priority, resumable, 6 GiB; completed
shard receipts remain intact. The operational inventory is `/home/werg/data/dgxops/mining-progress.json`.
Remaining work at this observation: 900 port-record-s3-v1 shards and 15 oversized whole-trajectory shards
(1.2–2.9 GiB each) from the static campaigns. The bounded identity sample contains coherent teacher-program and
program-IR IDs in the sampled rows; it does not prove whole-file identity or justify splitting a trajectory.
The oversized inputs remain assigned conversion work, with evidence in
`/home/werg/data/dgxops/oversized-trajectory-samples.json`; no admission or source deletion follows this audit.

The call-store root-ID collision was fixed in `948fbb40`: a run has a unique root call ID and a separate logical
`run_label`; seed derivation retains the logical label. Current `runIdentity` and the focused
`call-store-run-identity.test.mjs` regression cover the mechanism. Do not duplicate this completed repair.
