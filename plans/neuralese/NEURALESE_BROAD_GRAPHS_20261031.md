# Broad Neuralese producer/consumer graphs from public sources (2026-10-31)

Goal (owner, 2026-10-31): expand LFM3 Neuralese communication graphs beyond the few task shapes of the
`big2-stream` tranches, as far as feasible, with honest targets. Status and counts: §7.

## 1. What is wrong with the current communication share

The joint SFT mixture's communication graphs (~37% of rows, `/mnt/external/natlang-lfm3-data-20261028/communication/
big2-stream-*`) come from `port_record_calls` programs (`build-port-record-calls.mjs`) and static curriculum replays.
Their producer is almost always the root's **inline `nl` literal**: the constant instruction
`Do what item.request asks, using only item.evidence. ...` read by every child (`inline-site:` writes). The exported
span is a fixed authored sentence, so the producer learns nothing about the source and the consumer imports a
constant. The view/ask corpus and port-record data are artifact-diverse, but they reached training as single-node text
or as these constant-instruction pairs.

A genuine producer target is a value a call **computes from content only it sees** and that a later call **needs**.

## 2. Rules every graph satisfies

1. **Source withheld from the consumer.** Documents live in the program's folder (`docs/NN.txt`) or are passed by the
   root's code to a child without being printed. The root sees only the request and the file listing; the consumer
   (the root's answering turn, or a named answering child) sees the producers' results and nothing of the documents.
2. **Producer target = a gold-derived view of its own document**, never the payload alone: the child's complete
   `return_result` call is the supervised action (2026-10-26 "complete producer targets"); the converter places
   producer markers only around the authenticated returned value. Views are verbatim selections (dataset supporting
   facts/sentences/spans/cells, gold-SQL-referenced schema, gold-call-referenced tool signatures, dataset event
   templates) or the exact line `No relevant facts.` / `Not needed.` for distractors. Never teacher-written.
3. **Consumer target = the dataset's gold answer** (answer string, SQL, call list, summary, template list).
4. **Real dependency.** Edges are the authenticated `result:` (printed child result) and argument-read edges of the
   e2e miner (`authenticated_recurrence_context_view`); graphs are whole trajectories rendered jointly
   (`native_chain_render` → `native_chain_materialize` → `native_chain_assembly`), so gradients flow from the consumer's
   answer through the imported producer spans in one pass. Multi-producer fan-in is one graph.
5. **Own outputs only.** Prompts, documents (tool output / argument declarations) and runtime messages are context with
   zero weight; supervised tokens are the actions: root `eval`, each producer's `return_result`, the consumer's answer.
6. **Serving rendering.** Trajectories are produced by the reference replay through the current ts-host collector
   (`references.mjs` `referenceRow` → `admitRow` → `materializeNativeRows`), i.e. the runtime's real system prompt,
   tool schemas, opening declarations and cut-off pages, from a pinned ts-host build of a main commit
   (`build/<commit>`, recorded in every receipt). Graphs are rendered with `prompt_rendering='full'` and the recorded
   tool schemas. No reasoning is invented: children answer directly (`direct_answer`), root actions carry the
   collector's action notes (zero weight).
7. **Custody.** Upstream test/validation → `test`; train worlds pass `ProtectedWorlds.check` (registered evaluation
   packets, held-union and quality groups, decision holds, split reservations) or become omissions with the reason.
   Related worlds stay on one split via source groups: the question/row id, every document (`document:<sha>`), and the
   natural world (`wiki:<title>`, `sql-db:<dataset>:<db>`, `tool-schema:<sorted names>`, `loghub-system:`,
   `qmsum-meeting:`, `multi_news:`, `fetaqa-table:`). The e2e miner's assemble applies the held union again and the chain
   builder the current acquired-connected protection and decision holds. Source-review holds (`pendingSourceReview`)
   are honoured per dataset id.
8. **Licences** are recorded per case as provenance facts, never a filter.

## 3. Program patterns (one generic builder)

`ts-host/scripts/inline-curriculum/build-broad-graph-calls.mjs` turns source-agnostic cases
(`natlang.broad-graph-case/1`, produced by `scripts/neuralese_data/broad_graph_cases.py`) into curriculum IR. Each case
declares its documents, one view function (name, description, instructions), the gold view per document, the request
and the gold answer. The builder chooses (seeded per case) one consumer pattern:

| Pattern | Root code | Consumer | Edges |
|---|---|---|---|
| `fan-in/root` | per document `const note = await view(doc, request); console.log(note)` | the root's next turn returns the answer | `result:` printed child results, N producers → 1 consumer |
| `fan-in/child` | collects notes, `return await answer_fn(request, notes)` | a named answering child whose arguments are the notes | argument-read edges, N producers → child consumer |

Per-item calls are awaited named functions (owner idiom: per-item delegation, direct use of inputs, typed string
answers). Document order is shuffled per case; distractor documents are kept (their honest view is the "none" line),
capped so a graph stays within the trainer's token geometry.

## 4. Sources (first set)

| Source | Producer view (gold origin) | Consumer target | Fan-in | Groups |
|---|---|---|---|---|
| HotpotQA distractor | per paragraph: its `supporting_facts` sentences verbatim, else `No relevant facts.` | answer | 2 gold + ≤4 distractors | `hotpotqa:<id>`, `wiki:<title>`, `document:` |
| 2WikiMultihopQA | as HotpotQA (`supporting_facts`) | answer | 2–4 gold + distractors | `2wiki:<id>`, `wiki:<title>` |
| MuSiQue answerable | per paragraph: the whole paragraph when `is_supporting`, else none | answer | 2–4 gold + distractors | `musique:<id>`, `musique:seed:<hop>` (dev seeds keep the train hold rule) |
| SQuAD v1.1 train | per paragraph of the same article: the sentence holding the answer span (char offsets), else none; paragraphs containing the answer string elsewhere are dropped | answer | 1 gold + ≤4 same-article | `wiki:<title>` (S1 namespace) |
| QMSum specific queries | per transcript segment: the dataset's `relevant_text_span` turns verbatim, else none | gold answer | segments of one meeting | `qmsum-meeting:` |
| FeTaQA | per row block of the table: the `highlighted_cell_ids` cells as `row | column: value` lines, else none | gold free-form answer | 2–4 row blocks | `fetaqa-table:`, `wiki:<page>` |
| Multi-News | per article: the extractive oracle sentences (greedy ROUGE-1/2 against the summary, `code-computed:greedy-rouge-oracle`) | the summary | 2–5 articles | `multi_news:` |
| Spider train | per table DDL: the table with the columns the gold SQL references (`code-computed:sql-reference`), else `Not needed.` | gold SQL | all tables of the db | `sql-db:spider:<db>` |
| BIRD train | as Spider, request includes the evidence hint | gold SQL | tables of the db | `sql-db:bird:<db>` |
| xLAM 60k | per tool schema: `name(param: type, ...)` of the tools the gold calls use (`code-computed:gold-call-reference`), else `Not needed.` | gold call list (JSON) | all tools of the row | `xlam-tools:`, `tool-schema:` |
| LogHub 2k | per log window: the distinct event templates of its lines in order (dataset human labels) | the distinct templates over all windows, in first-seen order | 2–4 windows of one system | `loghub-system:` (whole systems held, as the view corpus) |

Queued (blocked for concrete reasons, §6): TAT-QA/FinQA (no raw copy with evidence annotations on DGX; TAT-QA reaches
training through directory-sources), NarrativeQA (not downloaded), QuALITY (no evidence annotation), KodCode
(tests → asserts view → solution: needs execution-checked targets), CodeSearchNet (docstring view needs a consumer task
that is not string matching).

## 5. Pipeline and storage

1. `broad_graph_cases.py build --source S --out DIR/cases/S.jsonl` (CPU, ledger low): adapters + split/custody.
2. `build-broad-graph-calls.mjs` → `DIR/ir/<source>/NNNNN.ir.jsonl`.
3. Replay (`replay-port-record-calls.mjs`, the generic reference replay) on the pinned build → turns/rows.
4. `e2e_mine.py shard/convert/assemble` → authenticated records/edges/chains corpus.
5. `build_chain_graphs.py` (batch driver over `native_chain_*`, LFM3 tokenizer, full prompt) → native graphs
   + receipts; manifests in `training/corpus-manifests/`, registry entry in `training/neuralese_corpora.json`.

Storage: `/mnt/external/natlang-broad-graphs-20261031/` (NVMe stays > 80 GB free).

## 5a. The chain-graph converter (audit D7; supervisor assignment 2026-10-31)

The coverage audit (`LFM3_DATA_COVERAGE_AUDIT_20261031.md`, D6/D7) found the big2 communication rows are one program:
one 2-node graph per edge repeated each producer ~23x, the 64-readers-per-payload cap of `E2E_DATA.md` was not
applied, and 89.9% of producer targets are one byte-identical eval program. `scripts/neuralese_data/build_chain_graphs.py`
replaces that conversion for every e2e source (big2 tranches, rest27, teacher mining, these broad graphs):

- **One graph per chain**: every decision of a trajectory is a node; producer spans and consumer imports at their
  recorded positions; consumers' answer decisions stay (not only the paging decision).
- **Reader cap per payload**: 64 reads per constant payload (`inline-site`, handover, soft-state); 512 per computed
  child result (`result` edges, payload computed from content only the child saw). Later reads stay ordinary text
  context in their node.
- **Identical-target dedupe**: a decision whose exact target was already supervised 64 times keeps no weight (it stays
  only when it produces a kept edge, then only its exported span is weighted); writers of computed child results are
  exempt (their decision is conditioned on their own document). Decisions with neither weight nor kept edge leave the
  chain. `--cap-state` carries the counters across tranches, so caps are global.
- A chain left without edges is written as an ordinary multi-node graph (`ordinary-<split>`), keeping its item variety.
- Exclusions: current acquired-connected protection (held/quality/pending groups), native decision holds, source-review
  family holds, and `--exclude-families` (rest27: `edit_stream`, `contract_diagnosis`, graded by the since-repaired
  oracles; `plans/neuralese/oracle-repair-20261028/old-oracle-admitted-sources.json`).
- Rendering: full serving prompt with its tool schemas; a chain whose full graph exceeds 40k tokens is re-rendered
  with the program-header prompt (recorded per graph); zero context weight. Never truncated.

Applied to: rest27 (`runs/neuralese-e2e-cloud-rest27-protected-20261015-v2`, 11.7k chains), big2 tranches 009-012 (records
present), 000-008 after re-extraction from the protected big2 archive (their extracted `records.jsonl` were removed after
the old conversion), and teacher-trajectory mining (audit item 6b).

## 6. Decisions and exclusions

- Identity views (a producer returning its whole document) are excluded: as a natlang call they teach copying input
  (the port-record builder records the same omission, `reconstruct-is-identity-as-a-call`).
- MuSiQue: paragraph-level support flags only; the held `musique_hops` sentence/decomposition family is not used.
- Multi-News views are code-computed, not human-selected; recorded as `code-computed:greedy-rouge-oracle`.
- Multi-hop distractors (HotpotQA, 2Wiki, MuSiQue) that contain the answer string are dropped: supporting-fact
  annotations are not exhaustive (source-review holds on `knowledge_evidence`), so such a paragraph may hold a needed
  fact and its "No relevant facts." view could be false.

## 7. Status

(filled in as the build runs)
