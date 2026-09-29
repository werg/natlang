# Task sources and offline trajectory conversion

Research and implementation checkpoint: 2026-09-28. Primary repositories/cards and actual sample rows inspected.
The first implemented pilot is below. The 2026-09-29 expansion adds QASPER, SciFact and non-file TreeDST edits; see [expansion log](DIRECTORY_EXPANSION_20260929.md) for current counts, holds, audit evidence and pipeline changes. Further recommendations remain research rather than imported volume.
Downloaded examples under `runs/source-research-20260928/trajectory-samples/` are research artifacts only;
`trajectory-sample-summary.json` records schemas, outcomes, tool counts and dataset-server row counts.

## Implemented pilot and pipeline connection

| Source | Native cases | Approved decisions after final 8,192-token audit |
|---|---:|---:|
| WorkBench | 8 | 16 |
| CommitPackFT | 12 | 36 |
| TAT-QA | 11 | 77 |
| MuSiQue | 12 | 12 |
| SWE-smith independent file creation | 8 | 16 |
| **Total** | **51** | **157** |

Artifacts: `data/teacher/source-backed/{train.ir.jsonl,static.results.jsonl,static.turns.jsonl,static.manifest.json}`.
The final rendered/audited pilot is `runs/source-backed-20260928/verified/sharp.ready.jsonl`, with sibling
manifest, audit and rejection ledger. All 157 rendered decisions pass; maximum length 6,356 tokens,
8,442 supervised tokens. Synthetic action notes are masked as context. Thirty-two scripted direct result
decisions are retained as evidence but excluded from default reasoning SFT. In particular MuSiQue currently
contributes retrieval decisions; model collection is needed for reasoning-and-answer demonstrations.

Source acquisition and native replay made **zero model calls**. Rendering/auditing used a CPU-only container
and a tokenizer-only Spark profile configured with the serving Sharp template and `terse=false`; no weights
were loaded. This pilot's ready artifact is bound to that tokenizer profile. Production stages render and
audit again with their selected student tokenizer; no training stage was launched.

Both training entry points discover the ready static manifest:

Use freshly built main `ts-host` or a new training freeze for these builders. Existing generation workers
continue using v10; that older snapshot does not contain the new static-bundle helper/admission gate.

- `scripts/build_lora_sft.sh` validates it and appends its results to normal admission/materialization.
  `NATLANG_STATIC_BUNDLE=MANIFEST` selects a bundle; `=off` disables discovery.
- `scripts/create_training_pipeline.py` adds a frozen-runtime `validate-static-sources` stage feeding the
  native coding lane before preparation, rendering and token audit. `--static-bundle` selects another bundle;
  `--no-static-bundle` omits it. A generated, unlaunched recipe is `runs/source-backed-20260928/recipe.json`.

Rebuild from the repository root:

```bash
python3 scripts/acquire_directory_sources.py --limit 12 --trajectory-rows 40
npm --prefix ts-host run build
node ts-host/scripts/inline-curriculum/build-source-backed.mjs \
  --cache vendor/directory-sources --out data/teacher/source-backed
```

The trajectory pilot captured 120 rows, including 48 parsable known-success runs. It imports eight bounded
SWE-smith **file-creation operations as independently specified tasks**, with original payloads and source
success observations. Native request reads and typed terminal returns are explicitly marked as wrapper steps.
Full SWE issues converted: **zero**. Shell/install/test dependencies remain unsupported, and Nebius needs a
pinned task/base-commit join. NVIDIA's first 40 rows have unknown success. No source reasoning or delegation
was invented. Repository license text was checked at SWE-smith's source commit against its classified blob.

Quality controls: original train partitions; all matching WorkBench sender records retained; explicit change
requests for underspecified commits; TAT-QA evidence reads precede calculations and inconsistent/unsupported
derivations are rejected; MuSiQue excludes 2,768 development seed IDs and groups related paragraphs/seeds.
Converted replay evidence binds task, outcome and trajectory digests. Attribution survives both renderers.
Rejection/audit JSONL files accompany the bundle. No pending annotation gold was rewritten.

Validation: Node/browser build, browser type checks, 41 focused Node tests and 30 focused Python tests pass;
actual source acquisition, replay, materialization, Sharp rendering and token-exact audit completed.

## First choices

1. **Offline trajectory pilot:** Nebius OpenHands and SWE-smith's structured `tool` split. Measure portable,
   successful coverage before building a large adapter. NVIDIA Open-SWE-Traces follows for model diversity.
2. **New directory tasks:** WorkBench for state changes, CommitPackFT for document/config edits, TAT-QA
   for table-plus-document reasoning. MuSiQue adds evidence chains and annotated unanswerability.
3. **Later:** QMSum needs a reviewed summary oracle; FinQA needs a unit/answer consistency audit.

Existing published Workspace-Bench and MuDABench cases stay evaluation-only, following
[DIRECTORY_REDUCERS.md](DIRECTORY_REDUCERS.md). The new pilot is in the static data path, not the live teacher queue.

## Existing trajectories: potential to avoid teacher generation

| Source | Contents and terms | Conversion assessment |
|---|---|---|
| [Nebius SWE-rebench OpenHands](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories/blob/main/README.md) | 67,074 complete traces; card reports 32,161 successful traces. Roles, tool arguments/observations, final patch and `resolved`. CC-BY-4.0; preserve underlying repository rights. | First pilot. Structured editor calls are promising. Sample has 185 messages, 44 editor calls, 44 shell calls and `resolved=0`: exclude it from positive SFT. |
| [SWE-smith trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories/blob/main/README.md) | Dataset server reports 24,100 `tool` rows; XML/ticks formats also exist. Serialized messages, patch, instance ID and success. MIT card. README's old 5,017 count does not describe current uploaded splits. | Prefer `tool`; deduplicate alternative formats. Sample has 49 messages, 12 editor calls, 10 bash calls and a successful verdict. Tasks need the original injected-fault environment. |
| [NVIDIA Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces/blob/main/README.md) | Messages, tool schemas, reference/model patches, repository license and success. CC-BY-4.0. v1.0 now has 151k traces after removal of git-hacking traces; v1.2 released Sept 26. | Pin one version/split. Inspected v1.1 OpenHands split has 80,630 rows. Sample has 113 messages and `resolved=-1` (unknown), so cannot qualify as a verified positive. |

All three inspected samples require package or repository execution: examples include `pip install -e .`,
`pytest`, a Python reproduction script and `python -m aiosmtpd`. A tool-name rewrite alone cannot preserve
these executions. Current evidence establishes substantial candidate pools, **not portable yield**.

### Offline adapter contract

Our task IR is `natlang.program/2` (`ts-host/src/teacher/program.ts`): a file-backed `.nl` root, callable
project files, inputs, starting `folder_files`, expected result/files and an explicit oracle. The native training
materializer additionally consumes decision contexts, actions/results, outcome and provenance. External chat
messages cannot simply be placed in `program_ir` or stamped with our collector's provenance.

Proposed conversion, without new model calls:

1. Pin upstream revision, instance, original environment/base commit, trajectory ID, model, terms and split.
   Join the corresponding task source and reconstruct every file/artifact needed by the trace.
2. Filter for explicit task success. Keep the original success label separate from our conversion validation.
   Failed and unknown runs remain research/repair material, not positive demonstrations.
3. Wrap the original task in a directory-reducer root. Map editor view/create/replace/insert operations to
   native file operations only where range, uniqueness, errors and observations retain their meaning.
   Map terminal submission to our typed result with a checked final workspace.
4. Replay translated actions with a deterministic driver through the runtime. This requires no teacher
   generation. Preserve original action order and evidence; regenerate native observations from actual execution.
   Reject semantic discrepancies rather than copying an incompatible observation as if our tool produced it.
5. Unsupported shell/test/install/network actions exclude the whole case unless a complete independent slice
   can be validated with its real starting state and all needed context. Do not erase an unsupported middle
   step and retain dependent later claims. Do not fabricate reasoning or delegation.
6. Check replayed files against the successful source run's model patch, plus the source task's success evidence
   where reproducible. Matching a patch validates conversion fidelity, not independently the patch's correctness.
   Record source verdict and native replay/file checks separately in an explicit external-source provenance path.
7. Deduplicate across versions, formats and derivative datasets. Group repository/task/related source material
   before splitting. Exclude reserved benchmark cases and overlapping held-out task families.
8. Respect the student's 8,192-token limit. Only create slices with reconstructed pre-step state and sufficient
   context; otherwise reject oversized traces. Retain useful failed attempts followed by verified repair inside
   a successful trace. Do not arbitrarily truncate histories.

Pilot report should count success-filtered rows, environment joins, supported tools, replay agreement, length
fit, deduplicated tasks and final admissions. Stop reasons should identify the unsupported operation or missing
artifact. The first pilot is a coverage audit; implementing new execution facilities is a separate decision.

Imported traces will largely teach navigation, edits and repair. They do not automatically supply natlang
delegation examples. Maintain some native generation with optional direct/delegated/mixed solutions and our
existing three-layer ad hoc limit; never invent child calls in a source trajectory.

## New tasks with usable gold

| Source | Available evidence | Directory adaptation and admission |
|---|---|---|
| [WorkBench](https://github.com/olly-styles/WorkBench) | 690 tasks, five sandbox databases, 26 read/write tools; MIT repository. Templates, initial CSV assets and expected operations/state. Use corrected v2 outcomes. | Represent email/calendar/CRM/project state as files. Validate expected changes and untouched state. Sending becomes a simulated outbox file. Group templates/workspaces into held-outs; no official train partition. |
| [CommitPackFT](https://huggingface.co/datasets/bigcode/commitpackft/blob/main/README.md) | 702,062 filtered commit records with instruction-like messages and before/after content. Markdown 62,518 + YAML 114,320 + JSON 39,777 = 216,615 raw candidates. Card MIT; original repository licenses vary per row. | Select clear self-contained edits with eligible original licenses. Group by repository/commit. Multi-file cases need pinned sibling files, not invented context. This is before/after data, not agent trajectories; deterministic read/edit/write demonstrations are possible for unambiguous edits. |
| [TAT-QA](https://nextplusplus.github.io/TAT-QA/) | 16,552 questions over 2,757 table/text contexts; answers, derivations, scales and relevant paragraphs. Dataset CC-BY-4.0. | CSV tables plus document notes; return answer, scale and evidence IDs. Check numeric units and source spans. Use original train only; group related contexts. Stronger immediate numeric source than unaudited FinQA. |
| [MuSiQue](https://github.com/stonybrooknlp/musique) | Multi-hop questions, support paragraphs/aliases; Full includes annotated unanswerable cases. CC-BY-4.0. Authors supply seed IDs to detect cross-dataset dev/test leakage. | Article folders with an answer/evidence manifest; score answer and support coverage. Use genuine annotated unanswerability. Exclude held-out seed IDs and group shared paragraphs across partitions. |
| [QMSum](https://github.com/Yale-LILY/QMSum) | 1,808 query-summary pairs over 232 meetings, transcripts and relevant spans; MIT repository. | Meeting evidence reports. Summaries require reviewed judging; action-item extraction is not already gold-labeled. Keep related meetings together. |
| [FinQA](https://github.com/czyssrs/FinQA) | Financial tables/text with programs and support facts; [official dataset terms](https://finqasite.github.io/) CC-BY-4.0. | Numeric evidence tasks after auditing program/answer/unit consistency. Inspected `ADI/2009/page_49.pdf-1` has display answer `380` versus executable answer `3.8`; do not declare or repair a label until conventions are reviewed. |

WorkBench sample: deleting a sender's last email maps to an explicit email ID. Inspected revision
`49c7dfd00c03d384ec59ea57374f50b766aa5613`. TAT-QA train and QMSum train samples parsed successfully;
their schema supports the adaptations above, but this is not a full corpus quality audit.

## Sources to reserve or defer

- [Workspace-Bench](https://github.com/OpenDataBox/Workspace-Bench): 388 tasks with large file workspaces,
  rubrics and dependencies. Keep published cases in evaluation. Authors fixed rubric/metadata leakage in August;
  keep judging metadata outside the agent's workspace. Current repository LICENSE is MIT, unlike the Apache
  attribution in our older plan; dataset terms still need their own confirmation before any training use.
- [MuDABench](https://github.com/Zhanli-Li/MuDABench): 332 questions, 589 PDFs; dataset card Apache-2.0.
  Evaluation-only in our plan. Authors report coverage/unanswerability problems; audit available evidence first.
- [DocuBench](https://github.com/DocuPipe/DocuBench): current README lists 72 documents with schemas,
  labels and a field scorer. Code MIT, labels/schema CC-BY-4.0; original documents have per-record rights.
  Public availability is insufficient to infer training permission. Image/PDF tasks need faithful document views.
- [CRMArena](https://huggingface.co/datasets/Salesforce/CRMArena/blob/main/README.md) and
  [ITBench trajectories](https://huggingface.co/datasets/ibm-research/ITBench-Trajectories): noncommercial
  terms and substantial environment dependencies; defer from the general import path.
- [AgentSynth](https://huggingface.co/datasets/agentsynth/agentsynth-trajectories): inspected row is generated
  by `mock`. Its verified flag checks argument shape and safety, not task achievement; generic observations
  and arithmetic fail to establish the requested customer metrics. Exclude such rows from quality positives.

## Next concrete work

Broaden the bounded trajectory scan for genuine multi-step portable edits; add pinned Nebius task/environment
joins before importing those traces. Collect reasoning trajectories on the new QA/workflow cases using the
existing teacher generation machinery. Expand source task coverage after reviewing pilot rejection ledgers;
do not infer full-trajectory compatibility from independently reconstructed file operations.


## Published expansion (2026-09-29)

`data/teacher/directory-expansion/static.manifest.json` now contains1,605 new cases
(CommitPack287, TAT-QA256, MuSiQue288, QASPER300, SciFact174, TreeDST300), after
excluding the51 pilot cases. Native replay approves4,119 decisions;1,049 remain held.
Final tokenizer audit admits all4,119. Combined with the earlier integrated static
preview:9,114 unique decisions, zero rendered duplicates or holdout overlap,3,676
directory reducer decisions (40.33%) and600 non-file tree-edit decisions. These are
static-preview counts, not the eventual complete generation/training mix.

Both default builders include this expansion beside source-backed and recovered
bundles. The generated joint training pipeline enforces the user's25% directory/file
reducer target after final token/dedup/holdout audit. See the expansion log for source
terms, preservation, scientific span holds, named-tree/JSON scoring corrections,
streaming memory fixes and additional unimported tree-edit candidates.
