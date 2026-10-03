# S1: Data inventory and migration

Draft, 2026-10-03. Detailed plan for stage S1 of the [Neuralese programme](README.md), consistent with [S0 revision 2](S0_SPEC.md). S1 runs continuously. Its first deliverable feeds port training (S3, gate G1); its Natlang task adapters keep feeding S5–S7.

## 1. Goals

1. Every dataset in bgkit, Schnitzeljagd, Natlang and on the external drive gets a recorded disposition: migrated, queued for a named adapter, replay-only, or excluded with a reason. Nothing is dropped silently. This extends the existing natlang policy: "discover and retain all artifacts; exclusion requires a recorded reason" (`training/data_sources.json`).
2. Accepted data exists as two model-neutral products (§2): **port records** for reading and writing, and **natlang executable tasks** (`natlang.program/2`).
3. Model- and harness-specific conventions of the source projects are removed; content and supervision are kept (README decisions table).
4. Splits are closed over source groups across all corpora, near-duplicates are removed across corpora, and no example puts later observations or answers into a writer's input.
5. The existing natlang corpus (v13 and its sources) is migrated to eagerly typed code with explicit captures (S0 §11.4).

Out of scope: the crisp skill-authoring corpus (S2), soft-substituted programs (S5), and RL environments (S7). S1 supplies their raw material and records the handoffs (§9).

## 2. The two products

### 2.1 Port records

A port record is a model-neutral example for training and evaluating the read and write ports outside full program execution. It holds text and structure only, never tokens, embeddings or chat-template markup.

```json
{
  "version": "natlang.port-record/1",
  "id": "bgkit:qa_squad:5733be284776f41900661182",
  "family": "qa_extractive",
  "task": "consume",
  "sources": [
    { "role": "document", "text": "Architecturally, the school has …", "exact_refs": [] }
  ],
  "writer": {
    "instructions": "Read the passage so that a later question about it can be answered.",
    "result_type": "Neuralese<Passage>",
    "context": []
  },
  "consumer": {
    "context": [ { "role": "user", "content": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?" } ],
    "withheld": ["sources"]
  },
  "target": { "kind": "text", "value": "Saint Bernadette Soubirous", "alternatives": [] },
  "contrasts": { "purpose_pairs": [], "distractors": [] },
  "outcome": { "label": "gold", "checked": "reference-answer" },
  "lineage": {
    "project": "bgkit", "store": "qa_squad", "store_version": "v1", "row": 0,
    "upstream": "rajpurkar/squad", "teacher": null, "sha256": "…"
  },
  "license": { "spdx": "CC-BY-SA-4.0", "noncommercial": false, "notes": "" },
  "split": "train",
  "split_groups": ["squad:article:University_of_Notre_Dame"]
}
```

- **`task`** is one of `consume` (a block is written from sources, a consumer answers with sources withheld), `reconstruct` (bootstrap and replay: the consumer reproduces source text), `continue` (the consumer continues a trajectory whose earlier part is in the block), `chain` (a block is written from earlier blocks plus new sources), or `compare` (several purposes or questions over one source, for purpose-sensitive writing).
- **`writer.instructions`** is the write site, where the purpose lives (S0 §4.4). It is derived from the consumer's actual use and the causally available context, never from the target. bgkit's per-slot "compression prompts" map here once their harness wording is removed (§4.1).
- **`writer.result_type`** is a natlang type, so a record renders directly into eagerly typed literal code.
- **`consumer.withheld`** names what the consumer must not see, so it has to use the channel.
- **`sources[].exact_refs`** marks identifiers, paths, quotations and numbers that must stay exact. Rendering keeps them as text next to the block, never only inside it.
- **`contrasts`** holds prompt and purpose contrasts, distractors and multiple questions over one source, so that S3 can pair purposes and compare correct, shuffled and zeroed payloads at matched length.
- **`outcome`** labels failed attempts and repairs as such. Only `gold` and `checked` targets are imitation targets.

Records do not carry budgets. Length is learned (README, "Length").

### 2.2 Natlang executable tasks

Sources with a checkable result, state change or effect become `natlang.program/2` records (`PROGRAM_IR_PIPELINE.md`) through new adapters. They then go through the existing collector, materializer and exporter with teachers. These tasks are the raw material for S2 skill authoring, S5 soft conversion and S7 RL. Conversion is selective: only where typed decomposition and checked results add value (semantics document §6).

### 2.3 From port records to rendered training examples

Port records stay model-neutral. A per-model renderer (S3) turns each record into:

1. a natlang program fragment in eagerly typed form, for example
   ```ts
   const passage: Neuralese<Passage> = <|neuralese|>⟦…⟧<|/neuralese|>;
   const answer: string = await nl`Answer the question using passage.`(question);
   ```
   rendered inside the backbone's native chat template with the native tool-call format;
2. the positions of the write-port block (producer side) and read-port block (consumer side), with the withheld sources removed from the consumer's view;
3. loss masks: the consumer's target tokens; the writer's stop decisions once generated sketches are used;
4. for bootstrap, the known-text embeddings that fill the block positions (port document §4).

Rendering happens only in that step; a change of template or tokenizer produces a new build and never edits records.

## 3. Inventory and dispositions

Counts are rows in the latest version unless noted. Older versions (`*_v1`, `*_v2`, …) are superseded duplicates and are excluded with the reason "superseded by vN" unless a later check shows unique content.

### 3.1 bgkit task stores

Location: `~/bgkit-data-nvme/bgkit2/tasks/` (201 files; parquet with identical `.arrow` mirrors). All stores share one text schema: `family, framing, context, prompt, instruction, tool_name, tool_args, target, split, meta`. In `tool_slots` framing, `context` is a JSON list of slot texts, `prompt` a parallel list of per-slot compression prompts, `instruction` the consumer's chat history as JSON messages, and `target` the next assistant message. `meta` holds upstream IDs and, for some stores, licence notes.

| Family (store) | Rows | Disposition | Notes |
| --- | --- | --- | --- |
| Extractive and multi-hop QA (`qa_squad`, `qa_hotpotqa`, `qa_hotpotqa_distractor`, `qa_triviaqa`, `qa_squad_article_v2`, `qa_narrativeqa`, `chatqa2_long_sft`, `chatqa2_narrativeqa`) | 98k, 98k, 47k, 58k, 8.5k, 14.6k, 30k, 14.8k | Port records, `consume` and `compare` | Group by article or document. Several questions per article become `compare` records. |
| Long-context probes (`qa_multineedle`, `qa_multineedle_long_fam`, `babi_filler_v3`) | 74k, 46.6k | Port records, `consume` | Synthetic filler; low weight. |
| `qa_quality_mcq` | 3.8k | Port records, `consume` | bgkit's 0.03 score was a format defect; re-derive the choice format from upstream. |
| Trajectory compaction (`compaction_v2`, `compaction_chunks_5src_100k`, `compaction_chunks_long_5src_fam`, `chunks_*`, `chunks_long_*`) | 40k; chunk stores 30–120k each | Port records, `continue` | Rebuild from the underlying trajectories (§3.3) at natural boundaries; drop bgkit's segment and spine structure. |
| Tool-output digests (`tool_digest_v2`, `tool_digest_ext_v3`) | 7.4k, 14.3k | Port records, `consume` | `tool_digest_ext_v3` has promptable summaries: good `compare` material. |
| Tool slots (`tool_slots`, `tool_slots_synth_v4_mapped`, `agent_tool_slots_v8`) | 40k, 3.3k, 90k | Port records, `continue` and `consume`; candidate executable tasks | Remove the harness: `zoom` calls, slot pricing, spine compaction, renamed tools. Rebuild from upstream trajectories where the mapping lost information. |
| Web browse-and-answer (`web_search_r1_v3`, `web_openresearcher_v3`, `web_openseeker_v3`, `web_webshaper_v3`, `web_sds_v3`, `web_miroverse_nc_v2`) | 50.8k, 72.9k, 19k, 6.2k, 1.6k, 59.7k | Port records; Natlang task adapter (offline web corpus as a service) | OpenResearcher and MiroVerse are CC-BY-NC derived: recorded, allowed. |
| Memory (`memory_recall_v2`, `memory_qa_v2_*`, `memory_qa_s2_ua`, `memory_agent_v2`) | 41.8k, 6.7k + 7.5k, 7.9k, 55.2k | Port records, `consume` and `chain` | `memory_agent_v2`'s two-stage index and search tools are bgkit machinery; keep the conversations and answers, drop the index protocol. |
| Repository context (`repo_qa_file_v3`, `repo_localisation_v3`, `repo_overview_v1`, `repo_tree_v1…v6`, `file_overview`) | 15.7k, 9.9k, 8.1k, 10.7k (v6), 150k | Port records; repository Natlang tasks (directory reducers over a repo folder) | Repo-disjoint held-out lists exist (`eval_repos_heldout.txt`, `train_repos_used.txt`, `*_heldout.parquet`). `repo_tree_v4*.roots.safetensors` are encoder outputs: excluded, not portable. |
| Coding agents (`code_swe_tsjs_v2`, `code_swe_opencode_v2`) | 25k, 8k | Port records, `continue`; executable only where an environment exists (§5.3) | Upstream nvidia/Open-SWE-Traces and Nemotron-SFT-SWE; MiniMax-M2.5 and Qwen thinking traces. |
| `agent_swe_lfm_v1*` | 2.4k (v1c); v2 is empty | Port records, `continue` | LFM2.5-8B teacher in bgkit's harness; strip harness. |
| Replay (`chat_replay_v2`, `read_replay_v2`, `read_replay_long_v1`, `reasoning_replay_v1`) | 75.6k, 35k, 6k, 12.6k | Replay-only | Ordinary-text replay against capability loss. |
| `babilong` | 11.9k | Held-out evaluation only | Eval-only upstream. |

### 3.2 bgkit benchmarks, corpora and raw data

| Item | Location | Disposition |
| --- | --- | --- |
| Benchmarks: benchpress, browsecomp_plus, hotpotqa_distractor, nq_open, squad, triviaqa_rc, swebench_lite, web pool and index; `showcase/` (babilong, ruler, longmemeval_s, locomo, musique, quality, narrativeqa, hotpot) | `~/bgkit-data-nvme/bgkit2/benchmarks/` | Protected held-out evaluation. Their upstream items are scanned out of every training corpus (§6.3). |
| Tokenized corpora (`corpus/{fineweb, code, books, dialogue}`) | `bgkit2/corpus/` | Excluded as stored (LFM2.5-350M token IDs). Rebuild from raw text (`filler/{fineweb_edu, pg19, wikitext}.jsonl`, `docs/raw`) as `reconstruct` records and replay. |
| Raw trajectories (agenttrove, openresearcher, open_swe_traces, nemotron_sft_swe_v35, swe_rebench_openhands, bench_dedupe) | `/mnt/external/bgkit-data/trajectories_ext/` | Upstream source for compaction, tool-slot and coding records; rebuild from here rather than from bgkit's mapped stores. |
| Web SFT (Search-R1, OpenResearcher, WebShaper, SDS, OpenSeeker, ASearcher) | `/mnt/external/bgkit-data/web_sft/` | Upstream source for web records and the offline-web Natlang adapter. |
| Teacher trajectories (`teacher/run1–3`, `fix1–4`, `lfm8b_swe_ov_v2`; `synth_*` on the archive drive, about 1M lines) | `bgkit2/teacher/` | Port records with outcome labels from their scores; failures kept as labelled feedback. |
| 12,083 bare git repositories | `/mnt/external/bgkit-data/repos/` | Source for repository tasks, file and repo overviews, and Git-history context. Tokenizer-independent. |
| `browse_trees/*.parquet` (git_commit_repro, git_history, kilt, lognav, babilong, xref) | `/mnt/external/bgkit-data/browse_trees/`, archive | Git-history and log-navigation sources for port records; KB-navigation framing dropped. |
| `qa_pairs`, `descriptions`, `structural` (per-repo LLM-generated) | `/mnt/external/bgkit-data/` | Port records after spot checks; teacher recorded as unknown where provenance is missing. |
| arxiv_v1, pubmed_v1, multi_news_v1, chatqa2_long_sft, swe_rebench, swe_repos | `/mnt/external/bgkit-data/` | Raw text sources for summarisation and `consume` records; low priority. |
| bgkit1 pre-tokenized data (`processed*`, `fineweb_edu_v1/tokens`, `mmap/`, `*_qwen_target`) | `/mnt/external/bgkit-data/`, archive | Excluded: Qwen3.5 or Falcon-H1 token IDs with no text; the same content is available from raw sources. Re-decoding is recorded as possible but not pursued. |
| Tree caches, survivor caches, `swe_embeddings`, checkpoints | `/mnt/external/bgkit-checkpoints/` | Excluded: bgkit model outputs, not data. |

### 3.3 Schnitzeljagd

Location: `/mnt/external/sdkb-archive/corpora/` (episodes as `episodes-{train,validation}.jsonl` with `query`, `answer`, `supports` (record texts), `required_ids`, `sufficient_groups`, `task_family`; plus `sources.jsonl` and `manifest.json` with SHA-256s) and `raw/`. Docs: `~/sdkb/docs/datasets.md`.

| Family | Episodes (train/val) | Disposition |
| --- | --- | --- |
| Recall probes: `recall-text` r2/r8, `parallel-recall` (+hard), `citance-recall`, `synth-people` r4/r32 | 3.8k/464; 4k/300; 3.3k/516; 15.1k/1.4k | Port records, `consume` and `reconstruct`. High-redundancy, high-entropy targets: the best material for the correct-versus-shuffled discipline. Inverse-cloze transcripts: excluded (retrieval-key training, which we do not do). |
| Text-to-SQL: `bird`, `spider`, `spider-memory` | 6.6k/1.5k; 8.7k/1.0k; 4.4k/560 | Port records (schema notes → block → SQL); Natlang task adapter with SQLite execution checks where databases are in `raw/agentic-20260927/{bird,spider}`. |
| Interactive worlds: `alfworld`, `scienceworld`, `webshop` | 5.7k/133/130; 4.5k/1.8k/1.8k; 1.6k/105 | Port records from recorded episodes; Natlang task adapter only where a simulator fixture can run (§5.3). |
| Tool use and function calling: `xlam`, `apigen-mt`, `agentbank`, `agentinstruct` | 58k/2k; 4.7k/275; 13.3k/807; 1k/54 | Natlang task adapters: schemas as typed services, decisions as checked returns (semantics document §6). |
| Code: `kodcode` | 68.6k/1.4k | Natlang task adapter with unit-test checks. |
| Reasoning: `knights`, `synlogic`, `reasoning-gym` | 6.1k/700; 20.5k/306; 5k/549 | Natlang task adapter where the generator has a checker; otherwise replay. |
| `*-bg` variants of the above | same | Same disposition; the background-knowledge framing becomes a `consume` record with the background as source. |
| Teacher datasets: `hermes`, `ultrachat`, `swe_smith`, `openhands` (prefix-memory and cross-experience protocols) | catalogued in `datasets.md` | Port records, `continue`; cross-experience episodes are handed to S2 as support/query material (§9). |
| SQuAD and HotpotQA bank corpora (`squad-*`, `hotpot-*`, `public*`, `r5/r6-mixed`) | various | Excluded as stored: four-space bank and retrieval framings. Their upstream questions are already covered by bgkit QA records. |

### 3.4 Natlang

| Item | Disposition |
| --- | --- |
| v13 model-neutral corpus (111,301 rows: 105,869 train, 5,432 test) and its producers (synthetic IR, source-backed static adapters, inline curriculum, failure-repair, directory reducers, self-improvement and optimizer slates, code corpus) | Migrated through the eager-typing and explicit-capture rewrite (§7). Remains the crisp backbone of the mix and the replay set. |
| `natlang.program/2` IR sources in `data/teacher/` | Unchanged as IR; re-collected only when runtime or prompts change. |
| Retired Python-pipeline sources (SCONE, Schema-Guided Dialogue, FinQA, CLEVR, sales, support) | Queued for Natlang task adapters (they have checkable results). |
| `data/external_pilot/` pre-IR traces | Excluded: not canonical (`PROGRAM_IR_PIPELINE.md`). |
| Development data on `/mnt/external/natlang-development-data/` (code corpora, ref-v7/v8, weakness sets) | Inventoried by the existing natlang ledgers; no change of disposition. |

## 4. Removing source conventions

### 4.1 bgkit

- **Harness text.** Remove bgkit's system prompts, tool names (`zoom`, `read`, `web_open`, `memory_*`), the `ANSWER:` format, the `<think>\n</think>` no-think convention and the `<|reserved_6|>` sentinel. Keep upstream system prompts only where they describe the task (for example an OpenHands task statement), recorded as consumer context.
- **Compression framing.** Per-slot "compress this to answer …" prompts become writer instructions phrased as the consumer's need. Ratios, `meta.slot_reps` and slot pricing are dropped.
- **Spine and segments.** Compaction's cons-cell spine and segment indices are dropped; compaction records are rebuilt at natural trajectory boundaries (turn or tool-call boundaries) from upstream trajectories.
- **Chat markers.** Any LFM tokens baked into text are removed; messages are stored as role/content structures.
- **Licence notes in `meta`.** Parsed into the `license` field; MiroVerse-derived rows are marked non-commercial.

### 4.2 Schnitzeljagd

- Drop bank, four-space, query-key and retrieval-specific fields (`required_ids` stays only as a provenance note on which supports are sufficient).
- "Use the stored notes …" query phrasing becomes ordinary consumer instructions, with the supports as sources.
- Manifests' `tokens: LFM2.5-350M` statistics are not reused.

### 4.3 Natlang

Natlang data is already model-neutral; its migration is the API migration of §7.

## 5. Natlang task adapters

### 5.1 Adapter contract

Each adapter emits `natlang.program/2` records: a root `.nl` function, its callable folder, `types.ts`, inputs, expected results or effects, folder files where needed, `source_groups` and `license`. Services model tools (search over an offline corpus, SQL over a fixture database, a world simulator). Exact paths, identifiers and source revisions are preserved. A source with missing environment state gets a fixture, a new validated teacher execution, or exclusion from executable training, never invented observations.

### 5.2 Adapters, in priority order

| Adapter | Sources | Check |
| --- | --- | --- |
| Text-to-SQL | bird, spider, spider-memory | Execute against SQLite fixtures; compare result sets. |
| Function calling | xlam, apigen-mt, agentbank, agentinstruct, hermes | Recorded call arguments match; effects through `record_args`. |
| Code | kodcode | Unit tests pass. |
| Offline web research | web_* stores and `web_sft/` with bgkit's web pool | Answer matches reference; search service over the offline pool. |
| Repository tasks | repo QA, localisation, overviews; bare repos | Directory reducers over a repo snapshot; localisation checked against gold files. |
| Reasoning with checkers | knights, synlogic, reasoning-gym | Generator's checker. |
| Retired natlang sources | SCONE, SGD, FinQA, CLEVR | Their original checks. |
| Interactive worlds | alfworld, scienceworld, webshop | Simulator fixtures, where they run in the host. |
| SWE | code_swe_*, swe_rebench, swe_smith | Test execution in containers; deferred until a container service exists. |

### 5.3 Environments

All execution environments are built: SQLite, unit-test runners, world simulators and SWE containers, each as a host service under `ts-host/src/services/` (or the equivalent location chosen in S4), with deterministic fixtures. Until an environment exists, its sources stay as port records only, recorded as "queued for adapter X: environment in progress".

### 5.4 Teacher generation

- Teachers: all available ones, including the paid teachers (Luna, Bunny) and local Qwen3.6-35B-A3B on the DGX. Local generation competes with training for the DGX and is scheduled against S3 runs.
- Each adapter starts with a pilot to measure acceptance rate and failure modes, then scales; volume is reviewed against learning curves rather than raw row counts.
- Accepted trajectories pass the existing collector checks; failures are kept as labelled repair material.
- Generation uses the post-migration runtime (§7) so that new trajectories are eagerly typed from the start.

## 6. Lineage, licences, splits and leakage

### 6.1 Lineage and licences

Every record carries project, store and version, row or episode ID, upstream dataset and revision, teacher (model and provider) where any, and a content SHA-256. Licences are recorded per record. Restrictive licences, including CC-BY-NC, are accepted (README decisions table) and flagged `noncommercial`. Gated sources keep their access terms in the ledger.

### 6.2 Split groups and closure

- Split groups are assigned before any windowing, chunking or episode construction: by article or document for QA; repository (and fork family) for repo and SWE data; trajectory and task instance for agent data (all attempts at one task together); tool schema group for function calling; database for text-to-SQL; generator seed family for synthetic reasoning; world instance for simulators.
- Groups are then closed transitively across all corpora: if any two records share a document, repository, task instance or schema group, they end up in one split. bgkit's repo-disjoint lists and Schnitzeljagd's group-first splits are inputs to this closure, not the final word.
- Upstream test splits stay test; nothing moves from test to train.

### 6.3 Protected held-out sets

- The natlang reserved evaluation prefixes (`s102:`, `s900:`) and protected aliases stay protected.
- bgkit benchmarks and showcase sets, Schnitzeljagd validation and test episodes, and a new Neuralese held-out set per family (sampled by group before training) are scanned out of training by ID, normalised text hash and near-duplicate match.
- A held-out family list for S3 consumer evaluation is fixed at the end of the first migration wave.

### 6.4 Cross-corpus deduplication

Exact deduplication by normalised text hash, then near-duplicate detection (MinHash over shingles) within each modality: documents, questions, trajectories, code. bgkit and Schnitzeljagd share upstreams (HotpotQA, SQuAD, OpenHands traces, SWE-rebench, xlam): duplicates keep the richer record and record the merge.

### 6.5 Leakage rules

- A writer's input contains only causally available context: the sources and what precedes the write site. Later observations, final patches, target answers and outcome labels are never writer inputs.
- `writer.instructions` are derived from the consumer's need and the causal prefix, never from the target text.
- Support sets for cross-experience and skill episodes never contain a query task's later solution.
- The consumer does not see withheld sources.
- A leakage audit checks each family on a sample by searching writer inputs for target strings and later-turn content.

## 7. API migration of the natlang corpus

Registered under `training/api-migrations/` as one entry covering S0's language revision.

1. **Eager typing.** A compiler pass inserts the checker's inferred types into model-written eval code: every `const` and `let` declaration, every inline `nl` binding, and every literal position. Unknown or open types are left open and reported; they are not guessed.
2. **Explicit captures.** A compiler pass rewrites inline `nl` calls that capture by name mention into `nl.with({ … })` with exactly the bindings the current compiler resolves. Explicit captures are snapshots by default (S0 §6); a captured `let` that the original call writes back is rewritten as `live(x)`, preserving write-back. Function-typed bindings are captured by value.
3. **Context semantics.** Programs are re-expressed under S0 §7 (callable folders as bound contexts). For existing programs this changes no behaviour; the migration validates that.
4. **Iteration.** `iterateOn` calls with TypeScript predicates get explicit bounds where they lack one; natural-language predicates are re-rendered under the new predicate prompt (S0 §13).
5. **Validation.** Each rewritten trajectory recompiles; checked results, effects and files are unchanged on replay; a sample is re-executed with the teacher under the new runtime.
6. **Builds.** The migrated corpus is a new snapshot; nothing built under the old runtime is appended to it.

The two rewrites also produce paired text and soft-function-literal material: every explicit-capture site is a candidate function literal for S5.

## 8. Validation

| Check | What it verifies |
| --- | --- |
| Schema | Every record validates against `natlang.port-record/1` or `natlang.program/2`. |
| Render | Every port record renders for LFM2.5-350M with the literal form; block positions, withheld sources and loss masks are consistent; nothing exceeds the context limit without being split at a group-preserving boundary. |
| Causal mask | No position in a writer block attends to consumer targets or later observations; checked on rendered examples. |
| Replay | Natlang tasks replay against their contract after migration and after every adapter change. |
| Leakage | §6.5 audit per family. |
| Split closure | No group appears in two splits after closure. |
| Dedup | Cross-corpus near-duplicate rate below a threshold (TBD) per family. |
| Coverage report | Per dataset: accepted, rejected (by reason), excluded (by reason); per family: train, validation, test counts; licence mix. |

## 9. Handoffs

- **S2 (skill authoring):** Schnitzeljagd cross-experience episodes, grouped Natlang tasks by family and support/query candidates; adapters' task families as the authoring domains.
- **S3 (port):** port records for the main families, the held-out family list, and the renderer contract (§2.3).
- **S5 (program training):** migrated natlang corpus with explicit captures; executable tasks with value-consumer traces for soft substitution.
- **S7 (RL):** executable adapters with their services and fixtures as environments.

## 10. Work items

1. Ledger: one dataset ledger covering all four projects (extending natlang's data inventory), with a disposition per item from §3.
2. Schema: `natlang.port-record/1` JSON schema and validator.
3. bgkit converters: QA and long-context; tool digests; web; memory; repository; coding; compaction rebuilt from `trajectories_ext`; replay. Each with convention stripping (§4.1).
4. Schnitzeljagd converters: recall probes; text-to-SQL; worlds; function calling; teacher-dataset prefixes.
5. Raw-text `reconstruct` and replay records from `filler/`, `docs/raw` and arxiv/pubmed.
6. Split-group assignment, transitive closure, protected held-out scans, cross-corpus dedup.
7. Leakage audit and coverage report.
8. API migration of the natlang corpus (§7), coordinated with S0's spec and S4's compiler passes.
9. Port-record renderer for LFM2.5-350M (owned with S3).
10. Natlang adapters in the order of §5.2, each with pilot and acceptance report.

Items 1–7 and 9 are needed for G1. Items 8 and 10 continue in parallel.

## 11. Exit criteria for G1 use

- Ledger covers every dataset with a disposition and reason.
- Port records exist and validate for at least: extractive and multi-hop QA, tool digests, compaction/continuation, web, memory, repository, recall probes, text-to-SQL, and replay.
- Render, causal-mask, leakage, split-closure and dedup checks pass for those families.
- Held-out families for S3 consumer evaluation are fixed and protected.
- Coverage report published.

## 12. Decisions on former open questions

Resolved 2026-10-03: all execution environments are built (§5.3); paid teachers are used alongside the local teacher; text `nl` keeps implicit live captures with `let` write-back (S0 §6), so write-back sites stay as they are and only explicit-capture literals use snapshots.

Also resolved: older bgkit store versions are excluded as superseded unless a check finds unique content; LLM-generated repository metadata (`qa_pairs`, `descriptions`, `structural`) is included with teacher recorded as "unknown".

No open questions remain.
