# LFM3-1.7B base on the held-v3 cadence subset (2026-10-27)

Owner request: quick evaluation of the newly released `LiquidAI/LFM3-1.7B` on natlang tasks.

- Model: `LiquidAI/LFM3-1.7B-GGUF@b3f2ae5d73cd3c91bf40e8ba752c648aa05dffdf`, `LFM3-1.7B-BF16.gguf`
  (sha256 `39158a79...5e9f68`). Architecture `lfm3`: conv + full/sliding(256) attention, partial RoPE on full
  attention, per-head sigmoid output gate, sandwich norm, 128k vocab, reasoning on by default, XML tool calls.
- Serving: upstream llama.cpp `d05706b` (has LFM3 model + template support; our fork and the vLLM image do not),
  `llama-server --jinja -np 16 -c 1048576`, server defaults (reasoning preserved across turns). Smoke: XML tool
  calls parse into OpenAI `tool_calls`, reasoning into `reasoning_content`; ~250 tok/s single stream.
- Harness: `scripts/held_task_budget_probe.py --spec-from training/neuralese/recipes/neuralese-direct-v2.json`
  (packet `natlang-held-task-packet-v3-20261012`, 16-case cadence subset), greedy, 7200 s / 1000 requests / 1000
  turns, 8 workers; the same path as the LFM2.5-2.6B GGUF reference in DECISIONS.md "Task-eval budgets".
- Placement: one-GPU CoreWeave sandbox `dfa72ee1` (RTX PRO 6000, 2 CPUs, 40 GiB), because DGX had 2.7 GiB free
  beside production training. Stopped after collection. Outputs: `runs/lfm3-1.7b-heldv3-20261027/`.

**Result: 13/16 task successes** (2 semantic_mismatch: case 0 workbench_tickets, case 16
relational_multihop_qualifier; case 5 edit_stream unresolved because the current runtime refuses it as a
quarantined curriculum case, `source_review_pending`, with zero model requests: 13/15 scorable). No budget
exhaustion, invalid action or repetition. Probe wall time 363 s; slowest task 350 s (case 32, 127 requests).

Comparison on the same subset: LFM2.5-350M base 0/16, LFM2.5-2.6B GGUF 5/16 (unbounded budgets), Mellum2.1 base
11-13/16. Base-model task outcomes only; no training or Neuralese capability is implied.

Native adaptation not yet done: the shared native port (`model/lfm2_port.py`, serving, GGUF export, tokenizer/role
markers) does not cover sliding-window KV splicing, partial RoPE, the attention output gate, sandwich norm or the
LFM3 tokenizer/tool format.

## Failure analysis (2026-10-28)

Read from the collected trajectories (`runs/lfm3-1.7b-heldv3-20261027/task-eval/probe/*/case-*/jobs/*.result.json`).

### The two wrong answers share one cause: exact or per-item work done in the model's own reasoning

- **Case 0 `workbench_tickets`** (expected T2, T3, T4; returned T3, T4). The program asks to judge each ticket's
  churn signal "one ticket at a time" (curriculum minimum sequence: narrow with crisp helpers, judge each item in its
  own `nl` call per semantic criterion, aggregate in code). LFM3 filtered dates correctly with
  `helpdesk.opened_between`, called the provided `ticket_kind` for kind, but judged the churn criterion itself inside
  one 19k-character root reasoning block. It missed T2 ("Two of our managers already asked me to look at other
  tools") and changed its mind on T3 between steps. It never issued an `nl` call for the churn criterion. Secondary
  errors on the way: `invalid-binding` (declared a variable named `ticket_kind`), six un-awaited NL calls
  (`[object Promise]`, "eval finished while 6 natural-language calls it started were still running"), and inside the
  `ticket_kind` child it tried to call `ticket_kind` again (`no-such-function`) instead of answering.
- **Case 16 `relational_multihop_qualifier`** (returned the correct 8 ids, wrong order). Paging, the current-edge
  filter (dropping `p-zang`, until 2023) and the union were all right. After its `for` page loop was rejected
  (`forbidden-loop`: not a checked counter loop) it fetched pages one by one, then sorted the ids by hand in reasoning
  and returned `p-paing` first. One `.sort()` in `eval` would have been exact.
- **Case 5 `edit_stream`**: not a model result; the runtime refuses the quarantined case (`source_review_pending`).

### Patterns across all 15 executed tasks (269 model requests, 204 evals)

| Pattern | Evidence | Consequence |
|---|---|---|
| Never uses natlang built-ins | 0 of 204 evals use `nl`, `iterateOn`, `decide` or `.sort()`; `for…of` 13, `.map` 6, `Promise.all` 5 | judgement/sorting in reasoning (both failures); loops as one request per step |
| Manual iteration | `actor_greenhouse`: 127 requests in one invocation, one heater/vent step per eval | succeeds slowly; would hit a 128-request budget |
| Binding collisions | `invalid-binding` in 10 of 15 tasks (redeclares inputs such as `schedule`, `from`, `level`; names results after services such as `facts`, `shipments`) | one wasted turn each; it always recovered from the error text |
| Guessing names for `read_code` | `no-such-function` 9 times in 5 tasks (`services`, `main`, `resolve.js`, `play_out`) | wasted turns |
| Async NL calls | un-awaited child calls in case 0 | lost results |
| Child calls over-think | child invocations: median 4,030 reasoning characters for one-line classifications; root steps median 341 | latency/tokens, no accuracy benefit observed |
| Tool-call format | all 269 replies were parsed tool calls; no text-only, malformed or repeated actions; no budget exhaustion | protocol is not the problem |

### Targets for the LFM3 recipe

1. **Natlang idioms in code** (highest value; both failures): SFT on verified teacher trajectories that delegate each
   per-item judgement to `nl` (awaited, `Promise.all`), compute filters/sorts/counts in `eval`, iterate with
   `iterateOn(...).until(...)`, and finish with `eval {finish:true}`. Weight the shapes where LFM3 currently works in its
   head: workbench/criteria reducers, multihop/argmax relational queries, actor/control loops, iterate shapes.
2. **Scope hygiene**: trajectories (and DPO pairs) that use inputs and service names directly; existing error texts
   already teach the fix, so RL on outcomes should also remove the wasted turns.
3. **Child-call behaviour**: inside a typed child, answer the declared type directly; do not try to re-call the
   function being executed; concise reasoning in child calls (measure live before changing any model-facing text).
4. **Whole-graph RL on the curriculum** (training split only, held packet stays protected): reward exact outcomes;
   these shapes give dense signal because LFM3 already reaches the answer most of the time.
5. **Measure before training**: repeat runs for noise; reasoning-not-preserved vs preserved; thinking on vs off;
   the full 64-row packet rather than the 16 cadence cases.
