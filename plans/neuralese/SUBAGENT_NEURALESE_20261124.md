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

The counts are in the results section below. No conversion is built.
