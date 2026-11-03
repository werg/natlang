# Learned-operator data: formats and converter specs (2026-11-03)

Design: `LEARNED_COMPACTION_OPERATORS.md`. Operators: `training/neuralese/natlang_neuralese/model/learned_operators.py`.
Every converter below is independent and self-contained: read §0 and your own section only.

## §0 Common contract (every converter)

**What you produce.** `natlang.port-record/1` records (schema `spec/neuralese-port-record.schema.json`, builders and
validator in `scripts/neuralese_data/records.py`) for exactly **one** learned operator, written through
`scripts/neuralese_data/operator_cases_common.py`:

```python
from neuralese_data.operator_cases_common import CaseSink, iter_jsonl, stable_rank
with CaseSink('<operator>', '<source-name>', converter=__file__, inputs=[...], caps={...}) as sink:
    for record in build_records():
        sink.offer(record)   # schema + operator rules + dedupe; rejects are written with their reason
```

Output lands in `/mnt/external/natlang-compaction-operators-20261103/cases/<operator>/<source-name>.{port-records.jsonl,
rejects.jsonl,manifest.json}`. You do **not** render, tokenize, admit, or edit `training/neuralese_corpora.json`; the
shared renderer (`natlang_neuralese/data/operator_graph.py`, owned by the operator agent) turns cases into native
graphs and applies the current protection union, decision holds and family holds.

**Record shape** (what the renderer reads):

| Field | Meaning for an operator case |
|---|---|
| `sources[]` | What only the operator site reads. Text sources: `{"role", "text", ...}`. History: one `{"role": "trajectory", "messages": [...]}`. |
| `writer.instructions` | The purpose the site compresses for, phrased from the consumer's need and never from the target (e.g. `Read this bash result for: <purpose>`). One string; for `extract` it applies to every document. |
| `writer.context` | Messages causally available at the site besides the sources (history: the task statement). Usually `[]`. |
| `writer.result_type` | A natlang type string, e.g. `Neuralese<ToolResult>`. |
| `consumer.context` | The consumer's own conversation up to its answer (it never contains the sources). |
| `consumer.withheld` | Always `["sources"]`. |
| `target` | The consumer's answer: the only supervised tokens (the model's own output). |
| `outcome.label` | `gold` or `checked` only (imitation targets). Teacher-written targets are rejected unless your section re-targets them with an exact code-derived answer. |
| `split` | Upstream test/validation → `test` (or `validation`); upstream train → `train`. Never move a test row to train. |
| `split_groups` | World keys via `records.group_key(namespace, value)`: the same namespaces the existing converter for that store uses (repo, task, document, wiki title, instance). Related worlds must share a key. |
| `lineage`, `license` | As the existing converter for that store sets them; record license as fact, never filter on it. |

**Seeds (owner correction 2026-11-03).** An operator writes a generated producer span that training starts as a
text summary related to the consumer's target (`LEARNED_COMPACTION_OPERATORS.md`, "Owner design correction"). Each
source may carry `meta.operator_seed = {"text", "provenance", "origin"}` with provenance `gold` (a dataset span:
supporting facts, reference summaries, answer sentences, relevant spans) or `checked` (exact code-derived text, e.g.
the refs a `:refs` target lists). Never invent a summary and never use teacher-written text as a seed. A source
without one gets the shared deterministic mechanical seed (`natlang_neuralese/data/operator_seeds.py seed_for`:
answer lines, question/target overlap sentences, identifiers plus outline, outline with head/tail, or a history
outline), filled by `CaseSink.offer` and, identically, by the renderer for older case files; the manifest counts
seeds by provenance. The seed lives only in `meta`: the site never reads it as input.

**Operator rules** (`operator_cases_common.RULES`, enforced by `CaseSink`):

| Operator | Sources | Roles | Chars per source |
|---|---|---|---|
| `compact-value` | exactly 1 | tool_output, file, document, passage, record, schema, tool_doc | 800 – 120,000 |
| `compact-params` | 1 – 8 (one site per source of at least 4,096; at most 110,000 in total) | document, passage, file, record, schema, tool_output | 200 – 120,000 each |
| `compact-history` | exactly 1 | trajectory | 8,000 – 200,000 (serialized) |
| `extract` | 2 – 8 | document, passage, record, file, tool_output | 200 – 20,000 each |

**Hard rules.** Never put the target or text derived from the target into `writer.*`, source text or
`consumer.context` (the only target-related text is the seed in `sources[i].meta.operator_seed`). Never read protected/held inputs as training data: keep each source's own held lists (bench keys,
`*_heldout` files, the S1 `final/` split closure) and leave the rest to the renderer. Selection under a cap is
deterministic: sort candidates by `stable_rank(record_id, '<source-name>')` and keep the first N per split. Stream
inputs (some files are 10-22 GB); never load a whole file. Put the converter at
`scripts/neuralese_data/operator_cases_<source_name>.py` with a `main()` taking `--limit` (smoke) and `--out-root`
(default `OUT_ROOT`). Add a small test at `tests/neuralese/test_operator_cases_<source_name>.py` over 2-3 inline
fixture rows (no large files). Run everything through the memory ledger at low priority, e.g.

```
python3 /home/werg/natlang/scripts/memory_ledger.py run --unit opcase-<source>-1 --budget-gb 6 --priority low -- \
  env PYTHONPATH=/home/werg/natlang/training/neuralese:/home/werg/natlang:/home/werg/natlang/scripts OMP_NUM_THREADS=1 \
  NATLANG_HASH_CACHE=/tmp/opcase-<source>-hc /home/werg/natlang/.venv-neuralese/bin/python \
  /home/werg/natlang/scripts/neuralese_data/operator_cases_<source_name>.py
```

Wait for the unit with an until-loop (`systemctl --user is-active <unit>.service`), then check
`systemctl --user show <unit>.service -p Result` and `journalctl --user -u <unit>.service`. Commit only your converter
and test (stage your own paths), pull, run `scripts/check-main.sh --quick`, push. Report: counts per split, reject
reasons, output sha256 (all in the manifest), and 3 sampled records' consumer question + target.

Existing closed S1 records (split closure, dedup and protected-set moves already applied) are at
`S1=/mnt/external/natlang-development-data/data/neuralese/port-records/full-20261003/resume-20261003T1948-review1/final/`
(`manifest.json` has per-family counts). Prefer selecting from them over reconverting.

## §1 Rendered graph format (renderer contract, for reference)

The renderer emits `natlang.neuralese-native-graph/1` rows plus a top-level `operators` list (see
`learned_operators.resolve_spans`). Per case:

* **Producer node(s)** `op0`, `op1`, … (one per source for `extract`, one otherwise). Rendered with the serving chat
  template through `operator_site_messages`: placement `system` → `system: <operator prompt>`,
  `user: <writer.context…> <writer.instructions>\n\n<source text>`; placement `user-suffix` (history) →
  `<writer.context> <source messages as conversation> user: <operator prompt>`. Then the assistant reply is
  `produce_open, <seed tokens, at most the operator's span cap>, produce_close`. `producing = [[seed_start,
  seed_stop]]`; the seed tokens and `produce_close` have weight 1 (annealed by the trainer's
  `operator_seed_schedule`), the site 0.
* **Consumer node** `consumer`: `consumer.context` with one consume-marked import per producer
  (`consume_open, <imported span>, consume_close`, joined by `\n`) at the start of its first user message, then
  `\n\n` and the original content; target reply weight 1, everything else 0.
* `operators`: `{operator, version, node, kind: 'prompt'|'seed', start, stop[, provenance]}` for each producer;
  `operator_conversion.seeds` records each seed's provenance, origin, tokens and clipping.
* **Text-cut sibling** (`text_cut_of: <graph id>`): the consumer alone, reading the runtime's current text cut of the
  source (`cutText`, 2,000 shown characters, head/tail with the `<<cut off: …>>` note) instead of the state. It is
  the baseline the operator must beat in the ablation.

## §2 compact-value ← bgkit tool digests (S1 `tool_digest` + four unconverted stores)

**Goal.** About 25k train cases where a tool result is compacted and a consumer must recover exact details from the
state alone.

**Inputs.** (a) `$S1/tool_digest.port-records.jsonl` (17,629 train / 4,063 test; family `tool_digest`, label
`teacher`); (b) bgkit stores not yet converted: `/home/werg/bgkit-data-nvme/bgkit2/tasks/{tool_digest.parquet (5,226),
tool_digest_v1.parquet (996), tool_digest_ext_v1.parquet (6,467), tool_digest_ext_v2.parquet (6,467)}`. Convert (b)
with `scripts/neuralese_data/bgkit.py:convert_tool_digest` (add the stores to `TOOL_DIGEST_STORES` with their generator
names `run0`, `run1`, `ext_synth_v1`, `ext_synth_v2`), reading parquet with `scripts/neuralese_data/common.parquet_rows`.
Dedupe (b) against (a) by normalized `sources[0].text` (`CaseSink` does this within one output; for cross-input dedupe
build the set of (a)'s text hashes first).

**Re-targeting (required).** The digests are teacher-written (`label: teacher`) and are not imitation targets. Keep
`sources`, `writer` and `split_groups`; replace `consumer` and `target` with an exact, code-derived recall task:

1. `refs = records.exact_refs_from(source_text, limit=64)` (paths, file:line locations, URLs, in first-seen order).
   Skip the row (reject `no-exact-refs`) when fewer than 3 refs.
2. Consumer question (user message): ``From the <tool> result read for "<purpose>", list the exact file paths,
   locations (path:line) and URLs it contains, in the order they first appear, as a JSON array of strings.``
   (`<tool>` from `sources[0].meta.tool`, `<purpose>` from the writer instructions after `for: `.)
3. `target = {"kind": "json", "value": [ref["text"] for ref in refs[:24]], "alternatives": []}`;
   `outcome = {"label": "checked", "checked": "code:exact_refs_from", "details": {"retargeted_from": "teacher-digest",
   "refs_total": len(refs)}}`; id suffix `:refs`; family `tool_digest_refs`.
4. Also emit, for rows whose source has at least 40 lines, a second case (id suffix `:lines`) asking
   ``How many lines does the <tool> result read for "<purpose>" have, and what are its first and last non-empty
   lines? Answer as JSON {"lines": n, "first": "...", "last": "..."}.`` with the code-computed answer (label
   `checked`, `checked: "code:line-stats"`). Same groups.

5. Seeds (checked): `:refs` cases carry `meta.operator_seed = {"text": "\n".join(the listed refs), "provenance":
   "checked", "origin": "code:exact_refs_from"}`; `:lines` cases carry the code-computed line count with the first and
   last non-empty lines (`origin: "code:line-stats"`).

**Caps.** Train ≤ 30,000 cases total across both case kinds (prefer `:refs`), test ≤ 2,000. Output source name
`tool-digest`. Expect roughly 20-30k train.

## §3 compact-value ← bgkit `web_search_r1_v3` (search results → gold answer)

**Inputs.** `/home/werg/bgkit-data-nvme/bgkit2/tasks/web_search_r1_v3.parquet` (50,779 rows; framing `tool_slots`;
`context` is a JSON list of result pages, `prompt` a JSON list of `web search <query>: <question>` strings, `target` a
JSON list of messages whose last assistant content is `ANSWER: <answer>`; `meta` JSON with `dataset`, `answer_match`).

**Mapping.** One case per row: `sources = [source("tool_output", "\n\n".join(pages), meta={"tool": "web_search",
"queries": [...]})]` (order preserved); `writer.instructions = "Read these web_search results for: <question>"`;
`consumer.context = [{"role": "user", "content": "<question>\nAnswer with a short phrase."}]`;
`target = {"kind": "text", "value": <answer without the ANSWER: prefix (records.strip_answer_prefix)>,
"alternatives": meta answers if present}`; label `gold`, `checked: "dataset-answer-em"`. Question = the text after
the first `: ` of the first prompt entry. `split_groups = [group_key("search-r1-q", sha256(question)[:20])]` plus
`group_key("wiki", <title>)` for every page title in brackets `[id] Title` if present. Reject rows whose answer string
does not occur (case-insensitive) in the pages (`answer-not-in-results`): the state must be able to carry it.

**Seed.** None supplied: the shared helper selects the answer-bearing sentences of the pages
(`mechanical:answer-lines`), which exist by the `answer-not-in-results` rule.

**Caps.** Train ≤ 15,000, test ≤ 1,000 (upstream test rows only; this store is all train, so carve test by
`stable_rank(question_group) % 100 < 3`). Source name `web-search-r1`.

## §3b compact-value ← question-driven tool-output cases (2026-11-03 rerun; floor and new sources)

**Floor.** `compact-value` has its own source floor, 800 characters (`RULES['compact-value']['min_chars']`; the other
operators keep theirs). The runtime cuts an output only past 2,000 shown characters, but real tool outputs worth
carrying exactly (a directory listing, a short file, a search page) are often 800-2,000 characters, and the operator's
state is also what a consumer reads instead of a value. Below 2,000 the text-cut sibling is the whole value, so such a
case teaches exact-detail carriage; the ablation baseline there is the full text. The reruns of `tool-digest` and
`web-search-r1` adopt the floor through `RULES`.

**Question-driven converters** (consumer asks a question whose exact answer is in the tool output; the seed is *checked*:
`operator_cases_common.question_selected_spans` = the sentences/lines that contain a gold answer, best question overlap
first, then more question-overlap lines; provenance `checked`, origin `code:question-selected-spans`):

* `web-<store>` (`operator_cases_web_qa.py`): bgkit `web_openresearcher_v3`, `web_openseeker_v3`, `web_webshaper_v3`,
  `web_sds_v3`, `web_miroverse_nc_v2` final rows. Source = the trajectory's tool results (search pages and opened pages in
  order, learned-compaction slots dropped, one `tool_output`); target = the `ANSWER:` text confirmed by exact match against
  the dataset gold and present verbatim in the results (MiroVerse finals are reports: the dataset gold answer of an
  `em`-flagged report). One case per question world over all stores; a question with any upstream-eval row is test
  throughout. v2 stores are superseded by v3 (same trajectories).
* `agent-tool-slots-repo-qa` (`operator_cases_repo_tool_qa.py`): bgkit `agent_tool_slots_v8` repo-QA final turns. Source = the
  one tool result (read/map/grep/list; `compaction` notes and the repo-overview marker are not tool results) containing the
  exact static-analysis answer; one case per task; repository/held-out/eval split closure as `tool-digest`; no question
  template above 30% of a split's cap.

Both reject a source text that the compact-value pool already contains (`duplicate-in-pool`; cross-converter repetition).
The repo-QA files overlap the compact-params S1 repo-QA sources only in repositories; the overlap count is in the manifest notes.

## §4 compact-params ← S1 long-document QA (argument listing of a large document)

**Goal.** About 25k cases where a call receives a large document argument and answers its instructions from the
compacted listing.

**Inputs (S1 final, already split-closed).** `repo_qa` (14,787 train, label gold `static-analysis`), `qa_narrative`
(13,046), `qa_narrative_long` (13,252), `qa_long_context` (10,860), `qa_mcq` (2,129), `qa_multihop` (89,660; use only
rows whose single source passes the 4,096-char floor), `qa_trivia` (44,476).

**Mapping.** Keep each record as is except: require label `gold`/`checked`, and rewrite `writer.instructions` into
the listing purpose, keeping the question: ``The call that receives this <role> answers: <consumer question>``
(the consumer question is the first user message of `consumer.context`). Keep `consumer`, `target`,
`split_groups`, `lineage`, `license` exactly.

**Multi-argument cases (coordinator request 2026-11-03).** A multi-passage record is a call with several large
arguments: keep every source, in order, one per argument (1 to 8 sources). The renderer gives each source of at
least 4,096 characters (the runtime listing's cut threshold, `SCOPE_VALUE_CHARS`) its own compact-params site and
shows shorter sources to the consumer in full, as the runtime listing does; the consumer sees one
`Compacted argument: <state>` entry per compacted argument. `CaseSink` rejects a case when no source reaches
4,096 characters (`no source reaches the compaction floor`) or when the sources total more than 110,000
characters (`sources total ... above 110000`, which keeps the rendered graph under the 32k-token cap). Measured on
S1 samples: `qa_long_context` rows have 2-7 passages of about 5-20k characters each, and most fit. `qa_narrative_long`
rows have 10-15 consecutive chunks of about 13k characters each (about 150k in total), so they exceed both the
8-source and the total limit. They stay rejected (`sources: N not in [1, 8]` or the total), because no subset can
be chosen without risking the answer's passage.

**Seed.** None supplied unless a family has a dataset-marked evidence span (then `gold`); otherwise the shared
helper's answer lines or question-overlap sentences (`mechanical:answer-lines` / `mechanical:overlap-sentences`).

**Caps.** Per family train caps: repo_qa 9,000; qa_narrative 5,000; qa_narrative_long 4,000; qa_long_context 4,000;
qa_mcq 2,129; qa_multihop 3,000; qa_trivia 3,000 (total ≤ 30,129); test ≤ 300 per family. One output per family:
source names `s1-repo-qa`, `s1-qa-narrative`, … (one converter script handling all families is fine).

## §5 compact-history ← SWE trajectories (S1 `trajectory_continuation_swe` + bgkit `compaction_v2`)

**Goal.** About 25k cases where an earlier stretch of a coding session is compacted and the next assistant message
is predicted from the task statement, the compacted state and the most recent steps.

**Inputs.** (a) `$S1/trajectory_continuation_swe.port-records.jsonl` (22 GB; 135,409 train / 29,930 test; built by
`scripts/neuralese_data/swe_compaction.py`, records already in the history shape: `sources = [trajectory messages]`,
`writer.context = [task]`, `consumer.context = [task, *recent]`, `target.kind = message`). Use only rows with
`outcome.label == "checked"` (resolved upstream; `teacher` rows are not imitation targets). (b) bgkit
`/home/werg/bgkit-data-nvme/bgkit2/tasks/compaction_v2.parquet` (40,003 rows; `instruction` = JSON message list of the
whole session, `meta.span = [start, stop]` the compacted stretch, `meta.trajectory_id`, `target` = the next assistant
text). Map (b) to the same shape as (a): `sources = [{"role": "trajectory", "messages": messages[start:stop]}]`,
`writer.context = [messages[1]]` (task), `consumer.context = [messages[1], *messages[stop:next_assistant]]`,
`target = {"kind": "message", "value": {"role": "assistant", "content": target}}`; drop the upstream OpenHands system
prompt as swe_compaction.py does (`_message`, `strip_markup`). Label: join `meta.trajectory_id` to
`/mnt/external/bgkit-data/trajectories_ext/swe_rebench_openhands/trajectories.parquet` (`trajectory_id`, `resolved`,
`instance_id`, `repo`): resolved → `checked` (`swe-rebench-tests`), else reject `unresolved`; trajectories not found →
reject `no-upstream`. Groups `repo:<repo>`, `swe-instance:<instance_id>`; bench instances
(`swe_compaction.load_bench_instances()`) → split `test`. Dedupe (b) against (a) by (trajectory_id, span).

**bgkit `compaction_v2` labels come from local test execution (2026-11-03).** Its 9,775 trajectory ids (UUIDs) are all
SWE-Zero trajectories (`/home/werg/bgkit-data-nvme/capability_packaging/trajectories_ext/swe_zero/data`; 6,091
SWE-Fixer-Train-110K, 3,402 SWE-Gym-Raw, 282 nebius/SWE-rebench; one trajectory per instance). SWE-Zero is execution-free:
there is no resolved status upstream, and none of the Open-SWE-Traces subsets contains these ids, so the OpenHands join above
finds nothing. `scripts/neuralese_data/swezero_eval.py` produces the labels by running tests:

- It applies each `model_patch` in the instance's environment and runs the harness's own eval script (SWE-rebench fork of
  SWE-bench for `rebench`, SWE-Gym's SWE-Bench-Fork for `swegym`), grading with that harness's log parsers and report code.
  The published instance images are x86_64 and the DGX is arm64, so each runs as a CoreWeave CPU sandbox image
  (`Sandbox.run(container_image=...)`; no GPU), repeating `run_instance` step for step.
- Instances without a published image are built in a sandbox from the harness's own base/env/instance Dockerfiles and
  setup scripts (`build_in_sandbox`); failed builds stay unlabelled.
- Every environment is validated by its gold patch (instance label stays `null` if gold does not resolve: its tests may not
  run at all) and by a no-patch control (never resolves).
- `labels.jsonl` (under `data_hdd/natlang-compaction-operators-20261103/swezero-eval/`): per trajectory id `resolved`
  true / false / null with `status`, `test_log_sha256` (logs kept beside it), harness repository and commit, image and
  sandbox provenance. `null` rows are rejected by the converter as `no-label-<status>`; only `resolved: true` is admitted
  (`checked: swe-zero-local-tests`, `local_execution: true`).
- SWE-Fixer-Train-110K publishes only patches, test patches and issue text (no FAIL_TO_PASS/PASS_TO_PASS lists, no
  environments). Its 6,091 rows stay rejected (`no-label-no-test-spec`); nothing is inferred from patch similarity.
  SWE-Gym-Raw likewise has null test lists for 3,288 of its 3,402 instances (only the 114 in validated SWE-Gym have them and
  published images). SWE-rebench-V2 (32,079 tasks, own multi-language harness) contains 12 SWE-Fixer and 43 SWE-Gym-Raw
  instances with test lists and images: a possible later extension, not evaluated.

**Seed.** The bgkit `compaction`/`compaction_v2` `target` is the next assistant message, not a compaction note, so
neither input has gold notes: the shared helper's `mechanical:history-outline` (actions, paths, errors, last step,
budget scaled to the stretch) seeds every case. A future source with actual compaction notes supplies them as `gold`.

**Caps.** Train ≤ 15,000 from each of (a) and (b) (≤ 2 windows per trajectory overall), test ≤ 1,000 each. Source
names `s1-swe` and `bgkit-compaction-v2`. Streaming (a) is mandatory (one line at a time, `iter_jsonl`).

## §6 extract ← broad cases (Neuralese-prompted extractor → parent that must use it)

**Goal.** About 30k fan-in cases: one extractor site per document, a parent answering from all extracts.

**Inputs.** `scripts/neuralese_data/broad_graph_cases.py` cases (`natlang.broad-graph-case/1`). The existing build
(`/mnt/external/natlang-broad-graphs-20261031/cases/cases.jsonl`) caps train at 5-6k per source; rebuild a larger
set into `/mnt/external/natlang-compaction-operators-20261103/broad-cases/` with
`broad_graph_cases.py build --out <dir> --sources hotpotqa,2wiki,musique,multi_news,qmsum,fetaqa,squad
--default-train-cap 8000 --test-cap 300` (the build keeps upstream splits, S1 group closure and source-review holds; check
`cases-build.json` omissions afterwards). Then convert every case whose `documents` has 2-8 entries.

**Mapping.** `sources = [source("document", doc["text"], title=doc.get("title")) for doc in case["documents"]]`
(order preserved; reject the case `short-document` when any document text is under 200 chars);
`writer.instructions = "Extract what this task needs from the document. Task: " + case["request"]["text"]` (never the
view's gold); `writer.result_type = "Neuralese<string>"`; `consumer.context = [{"role": "user", "content":
case["consumer"]["instructions"] + "\n\n" + case["request"]["name"] + ": " + case["request"]["text"]}]`;
`target` from `case["answer"]` with `kind` from `case["answer_kind"]` (`text` default); label `gold`,
`checked: case["oracle"]`; `split = case["split"]`; `split_groups = case["groups"]`; lineage project `upstream`,
store `case["dataset"]`, upstream_id `case["upstream_id"]`, revision `case["revision"]`; license
`records.license_(case["license"]["spdx"])`. Seed: each document's `view` (`meta.operator_seed`; `dataset-*` and the
`none-line` distractor line → `gold`, `code-computed:greedy-rouge-oracle` → `checked`). Skip task kinds whose answer is not a short text/choice (keep
`multi_news`/`qmsum` summaries: they are the summarizer cases; set `target.kind = "text"`).

**Caps.** Train ≤ 8,000 per source, ≤ 30,000 total (stable_rank by case id), test ≤ 300 per source. Source name
`broad-<source>` (one output per source).

**Not here.** The rest27 bgkit-knowledge graphs stay as text-prompted communication data in the corrected mix (they
are already rendered teacher-program graphs); their hotpot-derived knowledge is covered by `broad-hotpotqa`.
