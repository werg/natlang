# Subagent -> parent handoffs as Neuralese (2026-11-24)

Owner idea: "take subagent-heavy existing agentic training data and just turn the subagent to parent connections into
neuralese to train that in." Owner refinement the same day: "I'm not certain that the subagent training data needs to
be converted to Natlang. It could just be a source of diversity in our dataset that would still train neuralese."

## DECOMPOSITION: the mapping

Code: `scripts/neuralese_data/subagent_neuralese.py` (`build`, `scope`).

| Source part | Becomes | Loss |
|---|---|---|
| Lead (planner, main agent) conversation: system prompt, user task, its replies with tool calls, tool messages | One invocation (`lead`) of per-call rows. Each conversation is rendered by the serving engine's chat template for the student (`serve.targets.target_items` through `sft_sources.TemplateRenderer`): the dataset's own system prompt and tool schemas, its tool calls as structured `tool_calls` in the student's call format | Replies are targets. Prompts and tool messages are zero weight |
| Teacher reasoning (`<think>`) and text before a tool call | `reasoning_content` and lead-in of the reply | Zero weight (`sft_sources.think_context`, owner 11-24 "teacher reasoning as context") |
| Subagent conversation (its brief, its search/read calls, the recorded tool outputs, its final report) | One invocation per subagent (`sub:<session>`), rendered the same way. Its final report is its last reply. It is the producer of the value the lead reads | Its replies (actions and the report text) are targets. Tool outputs are zero weight |
| **The handoff**: the report text inside the lead's tool message (D01 `# Subtask k: … # Result: <report>`) | A Neuralese value written from the report's text. The tool message becomes content parts: text, `native_state` part, text. Nothing else in the message changes | Zero weight (context). Its gradient comes from the lead's later replies through the state |
| A dispatched subagent whose conversation the source did not keep (D01 keeps 3 per planner) | Its report is still a written Neuralese value the lead reads. The subagent has no producer rows | — |

How the lead reads a handoff (`--read-mode`):

- **`splice` (default): the learner's existing producer/consumer pass-through.** The lead's prompt carries
  `consume_open`, the report's written token ids as placeholder rows, then `consume_close`. An import sits at those
  rows. The source is the text-write producer document `[produce_open] + tokens` with span `[1, n+1)`. This is the
  same document `lower_rollout_turns.write_sources` reads from an actor's `block_write`, and its token ids equal
  `sglang_actor._sglang_write_text`'s (`_template_tokens(escape_specials(text))`). The learner computes the state in
  the same forward (`lfm3_fast_train.passthrough_plan`), so the lead's loss trains the producer computation, the
  embedding and the four marker rows (`--row-lr-scale`, text-initialised since 0462075ea). The rows go into chains
  through `exp0_sft.chains_of`, the format `sft_sources neuralese-graph` writes.
- **`inline`:** the written token ids inline, with no markers and no import. Since 2026-11-23 both engines read a
  text write this way by default (`engine.reads_in_context`, DECISIONS 2026-11-23 "Neuralese read-back"). In this
  mode the data is plain text and trains no Neuralese operator.

**Why `splice` is the default.** The owner's goal is training the Neuralese producer/consumer operators, and only
the spliced read exercises them. It also matches the opt-out serving path (`NATLANG_NEURALESE_TEXT_IN_CONTEXT=0`).
The mismatch is real, though: today's default serving reads a text write inline, and DECISIONS 2026-11-23 measured
the spliced read of a context-free write at 0/12 read-backs. This set is the training signal that could change that
number. Whether the runtime goes back to splicing is a separate decision.

**Not done, and why.**
- *Producer state computed in the subagent's own context.* The subagent's whole conversation would be the source
  document, and its report would be a produce span inside it. That adds 5-30k tokens of source per handoff, and the
  D01 leads read up to 22 handoffs. Nothing serves this form today: a produce span read by another call is read in
  context since d6b3218e7.
- *Producer markers as subagent targets* (`produce_open … produce_close` sampled around the report). No engine samples
  a produce span (FALLOW C6), and training markers as targets in plain-format chats would be model-facing.

## Admission and checks

- **Verification:**
  - D01: the source's own `item_f1` against its gold (depth = 1.0; width >= `--min-width-f1`, 0.9). Gold answers
    are not in the release.
  - D02: the release's gold answer must be contained in the lead's final answer (QA normalisation).
- **Held separation:**
  - the held closure by id, split group and source group (`sft_sources.held_closure`, every `held-eval*` role);
  - each question checked against the inputs of every held set (held-v3, held-v4 and reservations): exact
    normalized match, or at least half of its 10-word shingles occurring in held inputs (`held_texts`);
  - `held_eval verify` on held-v3 with the task ids and groups.
- **Repetition:** at most one bundle per distinct normalized question (`--max-per-question`). build.json reports the
  distinct questions and the distinct report states.

## Conversion to natlang programs: scoping only (owner 2026-11-24, part 2)

No conversion is built. The counts come from `subagent_neuralese.py scope` (read-only;
`/home/werg/data/sa2nl-20261124/scope/*.json`).

**(a) Checkable answer** is a gold answer or an exact checker. **(b) Complete evidence** means every tool call of the
lead and of every dispatched subagent has its recorded output, and every dispatched subagent's conversation is in the
release.

| Dataset | Items | (a) gold | Teacher's final answer contains the gold | (b) complete | (a)+(b)+teacher right | Gold occurs in the recorded tool outputs |
|---|---|---|---|---|---|---|
| D02 SearchSwarm | 6,732 | 6,732 | 4,721 | 661 | 429 | 5,830 (all items); 4,437 of the teacher-right ones |
| D01 WideSeek-R1 | 3,806 | 0 (the release has only the source's `item_f1`) | item_f1 pass 2,943 | 1,979 (1,696 depth, 283 width) | 0 without upstream gold | not measurable |
| D10 evidence-subagent | 10,379 | not in the row (parent question only) | no lead | 0 (the release keeps `selected_original_tool_events`, a subset) | 0 | — |

D02 by answer type, teacher-right and complete (429): short entity 359 (+1 with the gold absent from the evidence),
CJK short 41 (+2 without), number 12, CJK number 5, phrase 9. Most of D02 is "subagents missing": 80,147 subagents
dispatched, 41,300 conversations kept. D01 keeps at most 3 workers of up to 22+ per planner, so width items are rarely
complete.

**What a natlang conversion of that subset needs** (one real item per task, its own recorded evidence, no templates):
- **Program shape:** `research(question): string` (the lead) with a callable-folder child `investigate(brief):
  Neuralese<Report>`, where `Report = { findings: string; sources: { url: string; title: string }[]; open: string }`.
  The lead fans out with `Promise.all`. These are product functions written once per dataset, with the dataset's
  instructions as the natural-language policy.
- **Tool mapping:** a `web` service, crisp, outside world:
  - `search(query)` -> `web.search(query)`. D02 batches queries; the natlang call takes one query per call, and the
    model maps over a list;
  - `visit(url, goal)` -> `web.visit(url, goal)`;
  - `google_scholar` -> `web.scholar(query)`;
  - `PythonInterpreter` -> plain `eval` (TypeScript), no service;
  - `call_sub_agent` / `create_sub_agents` -> `investigate(...)` calls.
- **Fixture:** our policy issues its own queries, so a frozen replay of the teacher's exact calls is not enough. Per
  item, freeze a closed corpus: every search-result entry the trajectory saw (title, URL, snippet) and every visited
  URL's recorded extraction.
  - `web.search` ranks the item's own corpus (BM25) for any query.
  - `web.visit(url, goal)` returns the recorded extraction of that URL; an unrecorded URL answers "unavailable".
  - D02's visit outputs are goal-conditioned summaries ("The useful information in URL for user goal G …"), not
    pages, so a visit with a different goal gets the teacher's extraction. This is the main fidelity limit.
  - Assets: one JSON per item (sha-pinned, like the databench table service).
- **Checker:** the existing `span`/`qa` oracle (qa normalization with containment, polarity and core matching). CJK
  answers need a check that `qaAnswerText` keeps CJK characters. Numbers use `qaNumbers`.
- **Qualification per item:**
  - the gold occurs in the frozen corpus (above);
  - a reference replay of the teacher's own calls against the fixture reproduces the recorded outputs;
  - a negative control with an empty corpus fails;
  - the held closure applies, plus the held content check of `subagent_neuralese.held_texts`.
- **D01 needs upstream gold first:** WideSeek-R1's width tables and depth answers from its training-set release. Then
  width items need a table checker (row/column F1 with a threshold, as the source's `item_f1`).
- **Effort:** about 3 agent-days for D02. That is 1 day for the fixture builder and the BM25 `web` service, half a
  day for program files and task rows, 1 day for qualification (reference/negative replays and the CJK oracle
  check), and half a day for registry, exposure and pool promotion. D01 adds the upstream-gold acquisition and a
  table checker (+1-2 days). D10 has no lead and an incomplete evidence record: not worth converting.

## Results (2026-11-24)

**D01 WideSeek-R1** (`/home/werg/data/sa2nl-20261124/wideseek-splice`, registry
`sft-rows-sa2nl-20261124-wideseek-splice`, splice mode):
- Funnel: 3,806 trajectories -> 2,806 admitted. Dropped: 850 unverified (width item_f1 < 0.9), 125 without a
  handoff, 25 whose result message did not parse. Held: 0 by id/group, 0 by content.
- Contents: 8,955 chains (2,806 importing). 25,655 calls: 7,553 lead calls (4,747 of them consumers) and 18,102
  subagent calls.
- Neuralese: 42,143 reads over all calls (19,075 per chain, since a chain reads each state once), 18,932 distinct
  report states.
- Tokens: 46.7M forward, of which 3.1M are producer documents; 17.9M targets, 4.0M weighted (2.0M on lead calls; the
  teacher reasoning is masked). No call over the 32k limit.
- Quality (`quality.py` on chains.pkl): 25,563 distinct trained targets of 25,655; 0.8% of weighted tokens in exact
  duplicate targets; no marker token in any target. The most frequent trained 8-grams are the tool-call frames
  (`<function=search>`, `<function=create_sub_agents>`). Sampled targets are subagent reports and search calls.
- `held_eval verify` on held-v3 and held-v4 with the 2,806 ids and groups: no overlap.

**D02 SearchSwarm** (`/home/werg/data/sa2nl-20261124/searchswarm-splice-s{0..3}`, 4 disjoint shards by question,
registry `sft-rows-sa2nl-20261124-searchswarm-splice-s{k}`): see each shard's build.json. A single 500M-token file
was too large to build in a 14 GB claim and to load per round.
- On the first 16 bundles: 98 dispatches, 73 of them with a kept subagent conversation, and 15-142 handoffs per lead.
- Lead calls reach 61k tokens, so calls beyond the 32k limit are dropped (`too_long_calls`). They are mostly the late
  synthesis turns.

**D10 evidence-subagent:** not built. Its rows are child-only conversations with no lead, so there is no handoff to
make Neuralese.
