# Handover: training data, preference pairs, directory reducers (2026-09-27 evening)

> **Latest user authorization (2026-09-28): restart exactly one Luna repair worker.**
> This supersedes the earlier stop instruction. Use the current audited, deduplicated repair queue in
> `runs/luna-repair-20260928/`, with frozen runtime-v8, one collection worker and one model request in flight.
> Bonsai continues independently. Quality-pending, retired and unverified-contract cases stay excluded.
> Current worker PID 278800: 154 runnable repairs (149 task failures, 5 infrastructure-only), plus 3 legacy
> source cases needing reconstruction in deferred.jsonl. Nine other legacy tasks migrated successfully.
> Source shard is immutable while the collector runs; collector.log, jobs/ and summary.json record progress.
> Latest checkpoint: Luna 3 finished / 1 admitted / 2 task failures, 2 request-budget errors and 151 pending.
> Fixed frozen-world vendor paths and unhandled world-process spawn errors; Luna resumed from saved jobs.
> Bonsai remains on runtime-v7, queue-v5 (42 finished attempts / 517 pending); shared vendor mapping repaired.

> **Latest 2026-09-28 checkpoint:** use `runs/bonsai-recovery/queue-v5.jsonl` and frozen `runtime-v7`.
> It prioritizes the verified SMS repair handoff; 559 entries / 20 completed attempts / 539 pending at migration.
> Read DATA_QUALITY.md's fresh checkpoint. Token usage is now preserved in replay, fixing false handoff rejection
> from changed compaction timing. Folder guidance permits direct/delegated judgments and reuse of completed work.
> New output paths preserve interrupted extraction partials. Rewrite/answer-equivalence cases await independent
> review and are held out. Luna stays stopped; no training has started.

> **2026-09-28 data quality and slow-case review:** Bonsai now uses `queue-v4.jsonl` and frozen
> `runtime-v5`, with the existing append-only journal. New jobs have distinct paths; raw histories are preserved.
> Read [DATA_QUALITY.md](DATA_QUALITY.md) and the latest decision log before restarting or building data.
> The queue has 558 entries before journal skips: six distinct verified folder replacements, two measured FOLIO
> retries, and the filtered recovery work. Unsupported premise-removal and legacy CUAD probes are excluded.
> Inactivity (no saved reply for five minutes) is distinguished from hard deadlines; multi-item cases get larger
> bounded budgets. Quality-pending and partial benchmark results are held out of training/negative pairs.
> Correct direct/delegated/mixed results remain eligible. Luna stays stopped; no model training has started.

> **2026-09-28 full rejection audit:** Bonsai now uses `runs/bonsai-recovery/queue-v3.jsonl`
> and frozen `runs/bonsai-recovery/runtime-v4`, with the same append-only journal.
> Six reviewed probes precede the 546-entry recovery queue; completed attempts are skipped. Luna remains stopped.
> User's delegation policy: **three active ad hoc layers per root; every pre-existing file-backed `.nl`
> function starts a fresh root budget**. The third layer has a separate prompt/help variant and cannot create
> a fourth. Inline nl, Python nl and delegate share the limit. Correct direct and delegated answers can both
> be admitted. Execution policy provenance version 2 prevents silent reuse under the changed behavior.
> Full audit: `runs/rejection-audit-20260928/`; read `plans/REJECTION_REVIEW.md` and the latest decision log entry.

> **2026-09-28 rejection investigation:** Luna remains stopped. Current Bonsai queue is
> `runs/bonsai-recovery/queue-v2.jsonl` (546 eligible entries before journal skips), using
> `runs/bonsai-recovery/runtime-v3` and the existing `runs/bonsai-recovery/journal.jsonl`.
> Seven suspicious rejections replayed with correct answers: all belonged to retired `inline_type_repair`.
> Seven retired entries were removed from the queue; raw evidence remains excluded from training.
> Seeded handoffs now let a teacher replace the planted failure without reinjecting it. Seeded handoff
> provenance version 2 prevents reuse of incompatible old rows/partials. See the latest decision log entry.

> **2026-09-28 midday:** Luna is stopped at the user's request. Bonsai now runs the bounded recovery queue:
> `runs/bonsai-recovery/queue.jsonl` (553 unfinished cases), journal `runs/bonsai-recovery/journal.jsonl`.
> Runtime frozen at `runs/bonsai-recovery/runtime`; supervisor `scripts/run_bonsai_queue.py`.
> Limits: 600 seconds/case plus 10-second shutdown grace, 2 in-flight model requests, 128 new model requests/case.
> The old CUAD probe collector and launcher were stopped. Probe comparison remains incomplete and deferred;
> no file-tool default was changed. Read the latest decision log entry before restarting anything.

> **2026-09-28 morning:** user reduced Luna to **3 total workers** (folder 1, handoffs 1, redo 1).
> Active runner is now `/home/werg/natlang-generation-runner/ts-host`, with the streaming export fix.
> Bonsai train4 is complete; folder 102/106 saved, handoffs 282/833 saved, redo 12/13 saved at restart.
> Bonsai surface probe resumed at one worker after shared-context failures. See the latest decision log entry.

> Continuation 2026-09-27 night: see [GENERATION_DECISIONS.md](GENERATION_DECISIONS.md).
> Bonsai generation and **five total Luna workers** resumed (2 folder, 2 handoffs, 1 redo).
> This supersedes the two-worker limit below. Isolated runner refreshed; train4 restart now needs explicit reuse
> of `train4.resume.jsonl` (redo: `train4-luna.resume.jsonl`) with
> `--reuse-surfaces 58ec5990ab293f66d29b2a34115663a2371cf4b8fa5e10f3dc43ecdbb5b232d7`.
> Correction: current collectors publish after each job, but exports cover only their selected range. Use
> `node scripts/snapshot_teacher_jobs.mjs JOBS_DIR OUTPUT.jsonl` to gather all completed ranges.

Everything was shut down gracefully for a machine power-off: collectors finished their current step (partial runs are
journaled in each `*.jobs/` directory and resume on restart), the Bonsai server container (`natlang-bonsai`) was
stopped. Nothing is lost by restarting any collector below with the same arguments.

## Standing instructions (from the user)

- Kill processes by exact PID, never `pkill -f`/`pgrep -f`. Collectors: `ps -eo pid,args | awk '$2=="node" && /dist\/teacher\/cli.js/'`.
- Verify builds and tests before committing; commit freely. `while` stays forbidden in eval, bash and Python.
  Network is on by default in every tool. Browser parity is required; natlang runs in a worker in the browser.
- No junk data: delete backups, moved-aside dirs and superseded builds once their replacement is verified.
- Fix issues generally, not narrowly. Tool shapes are chosen by what our models do best (measured), nothing else.
- **One Luna repair worker is authorized by the latest user request.** Keep Luna at one collection worker and
  one model request in flight, including children. Bonsai continues the bounded recovery queue.
- Student: Sharp-Spark-X2.5-4B, trained at max-len 8192 (QLoRA, see `scripts/train_lora.py`, 8 GB GPU, 14 GB RAM).
  The user chose to keep generating before training.
- Another agent works in this checkout at times (it built the directory-reducer runtime). Check `git status` before
  committing; stage only your own hunks when files are shared.

## Machine layout

- Main checkout `~/natlang`. Runner checkouts, so collectors keep running while `ts-host/dist` is rebuilt:
  - `~/natlang-runner/ts-host`: rsync'd copy used by the Bonsai collector and the redo loop (older code: no handoff
    support). Refresh with `rsync -a --delete --exclude node_modules --exclude dist ts-host/ ~/natlang-runner/ts-host/`
    and `npm run build` there, only while no collector runs from it.
  - `~/natlang-folder-runner`: git worktree at `ddb136e` (current main), used for the folder batch and handoffs.
    Update with `git -C ~/natlang-folder-runner checkout --detach main && (cd ~/natlang-folder-runner/ts-host && npm run build)`
    while nothing runs from it. Remove it (`git worktree remove`) when no longer needed.
- Bonsai teacher server: `nohup scripts/serve_bonsai.sh > runs/bonsai-server.log 2>&1 &` (port 8081, 6 slots, 53k KV);
  wait for `curl -s 127.0.0.1:8081/health`. It uses ~7.7 GB of the 8 GB GPU; nothing else fits beside it.
- Memory is tight (14 GB; Bonsai's server takes ~6 GB RSS). Keep node scripts at `--max-old-space-size` 6000 or less.
- Times: `date`/`ps`/files are local (CEST); trace `observed_at` and some logs are UTC.

## Restart commands (all resume where they stopped)

```bash
cd ~/natlang && nohup scripts/serve_bonsai.sh > runs/bonsai-server.log 2>&1 &
# Bonsai: the 13 train4 cases still missing are all in 0-74
cd ~/natlang-runner/ts-host && nohup node dist/teacher/cli.js ~/natlang/runs/inline-curriculum/train4.ir.jsonl \
  ~/natlang/runs/inline-curriculum/train4.jobs ~/natlang/runs/inline-curriculum/train4.results.jsonl \
  --model-id Ternary-Bonsai-2-27B --root-seed 909 --server http://127.0.0.1:8081 --start 0 --limit 75 \
  --max-turns 20 --workers 6 >> ~/natlang/runs/train4.log 2>&1 &
BONSAI=$(ps -eo pid,args | awk '$2=="node" && /train4\.jobs/ {print $1}')
# Luna redo of Bonsai's rejected cases (1 worker)
cd ~/natlang && NATLANG_TS_HOST=~/natlang-runner/ts-host nohup scripts/teacher_redo_loop.sh \
  runs/inline-curriculum/train4.ir.jsonl runs/inline-curriculum/train4.results.jsonl runs/inline-curriculum/train4-luna \
  --while-pid $BONSAI -- --provider openai-codex --model-id gpt-6-luna --root-seed 909 --workers 1 \
  --context-tokens 16384 --max-turns 20 --reasoning-effort low --execution-plans >> runs/train4-luna-redo.log 2>&1 &
# Luna folder batch (1 worker)
cd ~/natlang-folder-runner/ts-host && nohup node dist/teacher/cli.js ~/natlang/runs/folder/gen1.ir.jsonl \
  ~/natlang/runs/folder/gen1.jobs ~/natlang/runs/folder/gen1.results.jsonl --provider openai-codex --model-id gpt-6-luna \
  --root-seed 1101 --all --workers 1 --context-tokens 16384 --max-turns 20 --reasoning-effort low --execution-plans \
  >> ~/natlang/runs/folder/gen1.log 2>&1 &
```

Paused (resume only within the 2-worker Luna budget, or on Bonsai with `--server http://127.0.0.1:8081` and
`--model-id Ternary-Bonsai-2-27B` instead of the provider flags, into a separate jobs dir):
```bash
cd ~/natlang-folder-runner/ts-host && nohup node dist/teacher/cli.js ~/natlang/runs/inline-curriculum/handoffs-v1.ir.jsonl \
  ~/natlang/runs/inline-curriculum/handoffs-v1.jobs ~/natlang/runs/inline-curriculum/handoffs-v1.results.jsonl \
  --provider openai-codex --model-id gpt-6-luna --root-seed 909 --all --workers 1 --context-tokens 16384 --max-turns 20 \
  --reasoning-effort low --execution-plans --reuse ~/natlang/runs/inline-curriculum/handoffs-v1.reuse.jsonl \
  --reuse-surfaces 8db6432281451c575b2d720c6ed120cd4a212d5a058139c9c71f1d6e6ab6c78e >> ~/natlang/runs/handoffs-v1.collect.log 2>&1 &
```
(The reuse file holds 57 handoff runs finished under the previous tool surface; the surface hash changed when inline
children's openings changed. A collector only reuses a finished result whose provenance matches, so after any change
to `src/native/*`, `src/scope-compiler.ts` or `src/environment.ts`, pass finished results with `--reuse` and the old
hash with `--reuse-surfaces`, or they are redone.)

## State of the data

| What | Where | State |
|---|---|---|
| train4 shard (967 cases) | `runs/inline-curriculum/train4.ir.jsonl` | 954 done; missing indices 0, 2, 4, 5, 10, 16, 30, 35, 49, 59, 64, 67, 68 (Bonsai range 0-74) |
| Bonsai results | `train4.jobs/` (128 results, incl. 36 in 150-449) | resume as above |
| Luna slices | `train4-luna-slice.jobs` (517), `-slice2.jobs` (264), `-slice3.jobs` (54) | complete |
| Luna redo of Bonsai rejects | `train4-luna.jobs` (6 of 10) | redo loop resumes |
| Handoff tasks (833) | `handoffs-v1.ir.jsonl`, `handoffs-v1.jobs/` (66 done) | paused |
| Folder batch (106 cases) | `runs/folder/gen1.ir.jsonl`, `gen1.jobs/` (30 done) | 1 Luna worker |
| Folder pilots | `runs/folder/pilot{1,2,3}.*` | done; pilot3 9/10, all admitted |
| Directory shards | `data/teacher/directory/{train,test}.ir.jsonl` (35 / 16 cases) | rebuilt with fixed families; test is held out |
| SFT set v4 | `runs/spark-lora/v4` (8267 turns, 3686 sequences) | built before corrections/pairs existed |

Collectors write `*.results.jsonl` only when they finish; before building from a run in progress, gather its
`*.result.json` files (one JSON per line) into a JSONL. For train4, the results are spread over four jobs dirs
(`train4.jobs`, `train4-luna-slice.jobs`, `-slice2.jobs`, `-slice3.jobs`) plus the redo `train4-luna.jobs`; gather all.

## What was built today (all committed on main)

- **Corrected variants** (`ts-host/src/teacher/corrections.ts`, `scripts/inline-curriculum/corrections.mjs`): where a
  call failed and then fixed itself, the same run with the fix made first, given the reasoning from before the failure.
  Checked by replay (control replay accepted, variant accepted, fix shows the same output, variant passes admission,
  reasoning does not name what failed). Only the fix is trained. v4: 159 variants from 276 sites.
- **Handoffs and preference pairs** (`src/teacher/handoff.ts`, `replay.ts`, `scripts/build-handoffs.mjs`,
  `scripts/build-preference-pairs.mjs`, `scripts/export-preference-pairs.mjs`): a teacher takes over a failed run at
  its failure; the teacher's decision there is preferred to the failed one. Replay matches calls by opening text or
  similarity; sites and pairs are checked by replay (the rejected response still fails in place). Handoff tasks are
  ordinary IR records carrying `handoff`; the collector replays the prefix. Old hard-state queue removed.
- **DPO trainer** `scripts/train_dpo.py`: continues the SFT adapter, reference = that adapter (log-probs precomputed),
  sides run one at a time with the exact split gradient (tested in `tests/test_train_dpo.py`, run in the
  `natlang-train` image). `--sft-split` holds out the SFT run's held-out programs.
- **`scripts/build_lora_sft.sh`** now also: runs corrections and appends the variant turns; writes `pairs.jsonl`
  (corrected pairs, plus handoff pairs with `NATLANG_HANDOFFS=runs.jsonl,...`) and `preferences.jsonl` (rendered).
- **Runtime fixes**: unawaited nl calls and dropped async callbacks no longer crash the host (callables mark promises
  handled; eval-realm rejections become host events); an inline child's opening says it is the judgment and that nl is
  not available (16% of Luna train4 runs hit the nested-nl refusal before).
- **Directory-reducer families** (the other agent built the runtime and six families; reviewed and reworked):
  agreement oracles (`oracle.ts`: `agreement` level, `semantics.files_oracle` with exact/rewrite/csv/counts),
  delegation required only from 100 files, real banking77 messages for `folder_mixed`, crisp label datasets only
  (sms_spam, sst2 whole sentences, banking77, clinc_oos), exact label names, CoEdIT drafts that need the edit (no
  paraphrase task), CUAD's own non-compete definition with quoted sentences, HotpotQA token-F1 with a short answer.
  Admission finds children by evidence too and matches JSON-escaped evidence. Luna pilot 3: 9 of 9 finished runs
  admitted (was 1 of 6 in pilot 1).
- **File tool surfaces** for a probe: `--file-tools all|editor|files` on the collector (`native/prompt.ts`), recorded
  as `provenance.file_tools`. Default (`all`) unchanged.

## To do, in order

1. **Finish train4**: Bonsai's 13 cases and the redo loop. Then build the next Spark set:
   gather all train4 results into JSONL files, then
   `NATLANG_TS_HOST=~/natlang-runner/ts-host scripts/build_lora_sft.sh spark runs/spark-lora/v5 <train3, train3.long30, simple-v1, train3-luna, ref-v1, ref-composed-v1 results> <train4 gathered> <train4-luna> <folder gen1 gathered>`
   (refresh `~/natlang-runner` from main first: it lacks corrections/pairs). Add `NATLANG_HANDOFFS=` with gathered
   handoff results for handoff pairs. Check the build log's corrections and pairs counts.
2. **Bonsai folder pilot** once Bonsai's train4 range is done: run `data/teacher/directory/train.ir.jsonl` (or a
   dozen cases from it) with Bonsai from `~/natlang-folder-runner`, inspect with the approach below. Bonsai is the bulk
   teacher; it may fail differently from Luna.
3. **Tool-surface probe** (the user wants the shape our models do best with): run the same small folder cases with
   `--file-tools all`, `editor` and `files`, each with Bonsai and with the untrained Spark student (serve its GGUF
   with `scripts/serve.sh`, not beside Bonsai: the GPU holds one). Compare acceptance, admission, tool errors and
   turns per surface; then decide the surface and, if it is not `all`, make it the default.
4. **Train Sharp-Spark** on v5 (only when the user says so): `scripts/train_lora.py` with the Spark options used for
   v4 benchmarks (`--model models/candidates/spark-x25-4b-train --trust-remote-code --load-in-4bit --no-kbit-upcast
   --optimizer paged-adamw-8bit --max-len 8192 --batch-tokens 8192 --target-modules q_k_v_proj,g_proj,out_proj,gate_proj,up_proj,down_proj`),
   then optionally `scripts/train_dpo.py` on `preferences.jsonl` with `--sft-split <run>/split.json`, and evaluate
   both on the held-out sets (simple/probe, and `data/teacher/directory/test.ir.jsonl` via `score.mjs`).
5. **On-policy loop after the first Spark run**: collect Spark on held-out/train cases, `build-handoffs.mjs` on its
   failed runs, collect the handoffs with a teacher, build pairs, DPO again.
6. **More folder data**: after gen1, larger builds (`build.mjs --families folder_triage,folder_index,folder_edit,folder_find,folder_extract,folder_mixed --seed <new> --shapes N`)
   into the regular queue. Known model pain point to address: models interpolate whole files into nl template
   instructions (`nl\`... ${text}\``) instead of passing the file; the runtime could point this out.
7. **Directory-reducer plan leftovers** (`plans/DIRECTORY_REDUCERS.md`): browser worker, document views, external
   trajectory conversion, Workspace-Bench/MuDABench adapters; the file-tool collapse waits on the probe.
8. **Old pipeline tests** `tests/test_student_improvement_pipeline.py` and `tests/test_student_improvement_runner.py`
   already failed at their setup before today (stale recipe infrastructure); fix or retire.

## How to inspect a run and iterate

- Per result: `outcome.accepted`, `outcome.oracle`, `outcome.files_check` (score, failed items), `outcome.detail`.
- Admission: `node ts-host/scripts/inline-curriculum/admit.mjs RESULTS.jsonl --ledger L.jsonl` prints rejection
  counts; each ledger line has `reasons` and `notes`. The standard SFT builder includes correct direct and
  delegated runs; use technique notes for coverage rather than treating direct answers as incorrect.
- Pattern that worked today: pilot a handful of cases per family, read the root's calls and what came back, and ask
  whether the task, its oracle or the runtime is at fault before blaming the model. Rebuild shards after family
  changes (`build.mjs` verifies every reference by replay and writes no shard if one fails).
