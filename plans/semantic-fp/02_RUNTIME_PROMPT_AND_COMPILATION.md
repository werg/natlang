# WS2: Runtime — model-facing prompt fixes, discovery, structured decisions, escalation policy, the optimization ladder

**Parent:** [Campaign index](README.md). **v2** (2026-11-23). v1's token-count goal is replaced by the design principles in §2.1.

## 1. Mechanism inventory

| Mechanism | Components | In scope |
|---|---|---|
| System prompt and call notes | `ts-host/src/native/prompt.ts` (`SEMANTIC_RESULT_PROMPT`, `SEMANTIC_CALL_COMPLETION_PROMPT`, `TOOLS_PROMPT`, `GENERATION_GUIDANCE`, `FUNCTION_TOOLS_PROMPT`, `directoryReducerPrompt`, depth-limit variants, `PROMPT_EXAMPLE_TEXTS`), `native/agent.ts:120-127` (`callNotes`), `native/system-prompts.ts` (piece registry, `Neuralese<SystemPrompt>` forms, DECISIONS row 40), `native/runtime.ts:270-325` (`read_code` builtin docs: `decide`, `iterateOn`), `native/runtime.ts:2329-2336` (`prompt-example` rejection) | all |
| Runtime teaching errors | `runtime/kernel.ts:526-545` (depth limit, `SAME_QUESTION_REFUSAL`, recursion), `native/return-errors.ts`, `native/runtime.ts` eval rejections, `spec/ERRORS.md` | all |
| Everything that renders or replays the prompt | `teacher/cli.ts`, `teacher/serving-request.ts`, `tests/neuralese/test_serving_prompt_rendering.py`, `scripts/neuralese_data/sft_sources.py` and `filled_reasoning` (render chains for SFT), the soft-prompt bank (`system-prompts.ts` piece IDs; trained `Neuralese<SystemPrompt>` forms ship in `.nz` banks) | R1's first step is to establish whether SFT renders own-success chains with the *recorded* system message or the *current* one. A trained target under a prompt it was not sampled with is "history that differs from what was sampled" (AGENTS.md model-facing rule). |
| Decision scoring | `native/decision.ts`, `model/decision-scoring.ts` (one place for all clients), `runtime/loader.ts:278-299`; engines: managed llama-server fork, wllama, Python native (`native_learning.SCORE_ITEM_FIELDS`), SGLang actor; guards `tests/neuralese/test_engine_parity.py`, `scripts/wllama_smoke.mjs` | R3: all four |
| Escalation | `fn.on({ model })` (exists), `runtime/runtime.ts:583-592` (`decide`), `runtime/pluggable.ts` | R4 |
| Optimization commands | `natlang compilations` (`specialize` is its alias, `cli/main.ts:261,338`), `natlang tune`, `natlang optimize`/`adapt` (`cli/adaptation.ts`), `natlang improve`, `natlang check --fusion`, `optimization/policies.ts` (pluggable search policies, default crisp), `calls/store-core.ts`, `calls/specializer.ts`, `compiler/rewrites.ts` + `DEFAULT_REWRITE_GATE` (`compiler/project.ts:205`), dashboard pages | R5 |

## 2. The model-facing prompt: first-principles fixes, no token target

### 2.1 Principles (replacing v1's "~400-token kernel")
The prompt is judged on what it teaches and on what it costs *after prefix caching*. A raw token count is not the
measure:
- **Caching is the cost model.** The system prompt and tool block are a fixed prefix that every engine reuses:
  SGLang's radix cache, the managed llama-server's per-slot prompt cache and `cache_messages` checkpoints, wllama's
  slot save/restore (ENGINE_PARITY, fork `cab27bfbe`), the Python native engine's prefix reuse. The learner reuses
  them too: shared prompt prefixes of ≥ 256 tokens are computed once as a prefix tree (`train/lfm3_fast.py`,
  `grpo.py`). A stable prefix's length costs one prefill per cache lifetime. Every distinct byte sequence before the
  first difference is a separate prefix, and everything after the first differing byte is recomputed, in serving and
  in training. Per-call assembly early in the prompt fragments the cache.
- **Stable core first, per-call parts last.** This is already the design: since d17a253fc (MODEL_FACING_CHANGES
  2026-11-12, "one constant opening for every call") the system prompt and tool list are constant. Program guidance,
  function-tools, folder and depth-limit text go in `callNotes` (`native/agent.ts:115-127`), the last paragraph of
  the call's opening user message. v2 keeps that invariant and adds a test for it (R0).
- **Gate a module only where it teaches better or the saving survives caching.** Moving text out of the stable
  system prompt into `callNotes` makes it per-call. It is then paid on every call that carries it, while the saving
  on calls without it is close to zero after caching. So gating is justified by teaching (the text misleads calls
  that cannot use it), or by a computed cache-adjusted saving (R0). Never by raw length.
- **Teach in errors, not prompts.** A rule that exists for one task shape goes. Where a check can detect the failure,
  the fix is taught by a one-sentence error at the moment it happens (AGENTS.md).
- **Say only true things.** Sentences that contradict the runtime are fixed.
- **No pressure against decomposition.** Child `nl` calls for distinct sub-questions are the paradigm.
- **Built-ins stay usable and discoverable wherever they apply.** `nl`, `nl.with`, `iterateOn` and `decide` apply in
  every eval scope, so the stable core describes them. Constructs that apply only to some calls are described in
  `read_code`'s builtin docs, plus a call note where the call can use them.
- **Every change is recorded in `plans/MODEL_FACING_CHANGES.md` and reaches every engine.** The prompt is rendered
  once in ts-host and sent to all four engines, so engine parity holds by construction. SFT and the teacher
  renderers are covered in the inventory row above.

*(v2 facts.)* `SYSTEM_PROMPT` = `TOOLS_PROMPT` (17,162 chars) + `GENERATION_GUIDANCE` (885) = 18,047 characters,
before tool schemas and call notes. `APPROACH_PROMPT` is **not** sent at runtime (teacher `--approach-guide` only).
The Neuralese sentences sit inside `TOOLS_PROMPT` unconditionally (~1.5k characters). `decide` is not described in
the system prompt at all, only in `read_code`'s builtin docs (`native/runtime.ts:270`), although it applies to every
call that can reach a function with a finite result. **Possible prefix break:** the 11-12 entry says `read_code`,
`edit_code` and `diff_code` are offered on every call, but `native/agent.ts:1276` today adds `edit_code`/`diff_code`
only when `session.codeEditable()`. If the tool block is rendered before or inside the system message, calls with and
without own code now have two different prefixes from the tool block on. R0 settles it.

### 2.2 R0: measure the cache-hit structure, and pin the stable prefix (1 day, no deps; first)
1. From real call records, compute the shared-prefix structure in tokens with the LFM3 chat template, per engine
   rendering. Sources: a recent DGX held run (`ops/held-v3/runs/`), an async round's collection, an
   `examples/triage` run, a VibeStudio capture. For each call, report the longest prefix shared with any earlier
   call in its episode and in its collection window, and where the first differing byte falls (system message, tool
   block, opening). Output: `ops/semantic-fp/prompt-cache/<date>.json` plus a table in the R1 audit doc.
2. Fix any break of the stable prefix that the measurement shows. The conditional `edit_code`/`diff_code` tool entries
   are the first candidate: offer them on every call (the 11-12 decision; a call without own code gets a
   one-sentence error), or move the difference after the stable part. Either form must keep L3's protected-item
   errors.
3. Add `ts-host/test/prompt-prefix-stability.test.mjs`. Render calls of different shapes (plain, own code, folder
   reducer, depth limit, Neuralese signature, decision readout, different return types) and assert that the system
   message and tool block are byte-identical across them. Only the opening message after its instructions may
   differ.
4. For every R1 move between the stable core and `callNotes`, R0's numbers give the cache-adjusted cost:
   per-call tokens added to the calls that carry the note, against the one-time prefix saving. R1 applies a move only
   when that cost favours it, or when the teaching argument does.

### 2.3 R1: one pass over every sentence (2 days, after R0; ships as the default)
An agent classifies every sentence of `SEMANTIC_RESULT_PROMPT`, `SEMANTIC_CALL_COMPLETION_PROMPT`, `TOOLS_PROMPT`,
`GENERATION_GUIDANCE` and `FUNCTION_TOOLS_PROMPT` into a table committed as
`plans/semantic-fp/prompt-audit-20261123.md`, and acts on it:

| Class | Action | Known instances |
|---|---|---|
| **Factually wrong or contradicting the runtime** | fix the sentence | "functions may not call themselves recursively" (becomes "only on a smaller argument" when WS1 L7 lands); `native/runtime.ts:1990` comment; any sentence about `edit_code` that L3 changes |
| **Missing description of a core built-in** | add one sentence | `decide(fn, ...args)` returns `{ value, probabilities, confidence }` for a function with a finite result. `.in(context)` gets one sentence in `read_code`'s builtin docs. |
| **Benchmark scar tissue**: a rule for one task shape, written for every call | delete it. If a runtime check can tell the failure apart, add a one-sentence error there; otherwise delete without replacement and let the held eval show any regression | `more: true` pagination (TOOLS and its depth-limit twin, `prompt.ts:99`); "save JSON and return it ... return exactly the JSON text saved in the file"; "When separate files pair a question or criterion with a state ..."; "For a conversation state, read the full exchange ..."; "A find/replace request applies to the matching span ..."; "An isolated quote can lose the scope ..."; "a recursive tree walker or an open-ended loop is unnecessary"; `change-request.json` (`directoryReducerPrompt`, `prompt.ts:80`) |
| **Wrong form spelled out** (AGENTS.md "never spell out a wrong form in a prompt") | rewrite as the positive form, and make sure a runtime error teaches the fix | `SEMANTIC_CALL_COMPLETION_PROMPT` ("Do not delegate the same question over the same unchanged input ..."). The runtime already refuses exactly that (`runtime/kernel.ts:534-537`, `SAME_QUESTION_REFUSAL`), so the paragraph is replaced by one positive sentence: "Give a child call a distinct sub-question and the inputs it needs; answer the question you were given here." "do not print a pseudo invocation"; "never copy a cut-off value" (the cut-off marker's own text already says where the value is); "do not redeclare or shadow those bindings" (`invalid-binding` teaches it) |
| **Anti-delegation pressure** | remove | The same paragraph's "When the evidence is already available, return the declared value directly", read together with the depth-limit notice, pushes against decomposition. Keep the pro-semantic sentence "If you get a question about meaning, don't try to solve it via code, keyword or regular-expression matching" in its positive form: "Answer questions about meaning by reading the data, or by mapping an nl function over it when it is large." |
| **Copyable domain example** | replace the refund-policy example with a schematic one in the call's own terms (`<criterion>`, `<items>`) as APPROACH_PROMPT already does; then delete `PROMPT_EXAMPLE_TEXTS` and its `prompt-example` rejection (`native/runtime.ts:2329-2336`): a guard against copying an example that no longer exists is dead code | `prompt.ts:13` (`'Refunds are allowed within 30 days of delivery.'`), `prompt.ts:124` |
| **Duplicate** | keep one copy | deep-copy guidance appears in TOOLS, GENERATION_GUIDANCE and APPROACH; large-input selection appears twice |
| **Capability text for some calls only** | decide per R0: either it stays in the stable core, or it moves to a call note at the end of the opening (`callNotes`, `native/agent.ts:120-127`) when it misleads calls that cannot use it or the cache-adjusted saving is real; the full text also goes in `read_code`'s builtin docs | the Neuralese sentences (~1.5k chars). The prefix is shared by every call of every program, so moving them saves nothing after caching. Moving them makes Neuralese-using calls pay ~350 tokens each. They move only if R0 or the audit shows that plain calls are misled by them (e.g. by producing Neuralese markers or wrapping text results). |
| **Correct and needed** | keep verbatim | `nl`, `nl.with`, typed holes, `iterateOn` (both forms, NL predicate), `return_result` statuses, cut-off markers, `transcript`, `compact_history` |

- **Rollout.** The new text is the default at once. `NATLANG_PROMPT_REVISION=2026-11-22` restores the previous text
  byte for byte (opt-out, removed after the next measured round shows no regression). Each changed sentence gets a
  MODEL_FACING_CHANGES entry: what changed and why, on first principles. No A/B gates the change (AGENTS.md "Decide
  model-facing changes on first principles"; owner 2026-11-18/22: held comparability across series matters less than
  progress). The next held measurement of the running expert-iteration lineage shows any regression. A pin that
  adopts it pairs across the series change with a warning. It does not re-measure a base.
- **Training interaction.** R1 lands on main. The async driver (`expert_iteration.py async-drive`) picks it up at its
  next pin, and collection then samples under it. If SFT renders own successes with the *current* prompt (inventory
  row 3), that is a pre-existing bug: fix it to render the recorded system message, with a targeted test, whether or
  not R1 ships.
- The trained `Neuralese<SystemPrompt>` forms (system-prompts piece IDs) are re-initialised from the new text for
  every changed piece ID. Unchanged pieces keep their forms.
- **Ordering.** Every addition to the stable core goes into the core text, never into a per-call position before
  it. No R1 change may put per-call content before the end of the call's instructions.
- Tests: `ts-host/test/prompt.test.mjs` (or the existing prompt tests) pin the new text; the opt-out reproduces the
  old text exactly; `prompt-prefix-stability.test.mjs` (R0) stays green; `test_serving_prompt_rendering.py` stays
  green.

### 2.4 R1b: one-sentence errors for each removed rule that a check can see (1 day, after R1)
For each deleted scar-tissue rule where the runtime can detect the failure crisply, add the error at the point of
failure. Examples: a finishing eval whose value is a `console.log` result; a returned string that is the file's
JSON but differs from the file; mutating a frozen parameter, where the error names the parameter and suggests
`structuredClone(param)`. Each error gets an `spec/ERRORS.md` row and a skill table row.

## 3. R2: library discovery at scale (2 days, no deps)
*(v2 fact)* The opening already switches to immediate callable interfaces above 96 entries, and `read_code` lists a
namespace and resolves a type alias (MODEL_FACING_CHANGES 2026-11-21). v1 §3 is partly built.
Remaining: `search_code(query)` as an eval builtin (lexical + signature search over names, parameter and return
types, `types.ts` aliases, doc comments, service declarations). It is documented in `read_code`'s builtin docs and
offered only when the scope has a codebase. The library-discovery task family (WS5) needs it. Tests:
`ts-host/test/codebase-discovery.test.mjs`.

## 4. R3: structured `readout: decision` on all four engines (4 days, no deps; one agent per engine pair)
*(v2 fact)* `finiteValues` (`native/decision.ts:19-37`) accepts literal unions and booleans through nesting,
refinement and `Untrusted`. `runtime/loader.ts` requires ≥ 2 values and forbids generic results.
- Build: finite records (all fields finite, product ≤ 32: score the joint options in one shared-prefix batch) and
  finite tagged unions. Above 32, score field by field: each field's options are scored on the shared prefix with the
  chosen earlier fields appended. All of it lives in `model/decision-scoring.ts`, so llama-server and wllama get it
  from one client. Python native and SGLang get the same score items. `SCORE_ITEM_FIELDS` and `test_engine_parity.py`
  guard it. `decide()` adds `fields?: Record<string, { value, probability }[]>`.
- Done when: `test_engine_parity.py` passes, `wllama_smoke.mjs` ranks a record decision like the native server within
  its 2.5-nat tolerance, and a Python-native vs SGLang comparison on 20 record decisions agrees on the argmax.
  Recorded in `docs/ENGINE_PARITY.md`.

## 5. R4: `escalate` is a natlang policy (2 days, after R3 for per-field floors; flat decisions need no dep)
Escalation decides which tier serves, which AGENTS.md names explicitly as policy. So it is a `pluggable`:
- `escalate(fn, { fallback: 'frontier', policy?: 'crisp' | 'nl' | 'shadow' })` in `@natlang/std/policy`. It runs
  `decide(fn, ...args)` on the default model, then asks the policy whether to escalate.
- **crisp** (default): confidence floor and per-value risk floors from options (`{ floor: 0.85, risk: { security:
  0.2 } }`).
- **nl**: `@natlang/std/policy/shouldEscalate.nl` (`readout: decision`, `boolean`). It reads the probabilities, the
  declared risk classes, the function's escalation history from the call store (rate, how often the fallback
  disagreed), and the cost of each tier. Its own successes and failures are training data like any other call.
- **shadow**: both run; crisp serves; `pluggable_shadow` records agreement.
- Escalation runs `fn.on({ model: fallback })` and records a `decision_escalate` trace event. `natlang profile`
  reads it (R5). The fallback's answers on escalated inputs are teacher demonstrations for that function's
  `natlang tune` and for WS5 (an escalation is a labelled hard case).
- Tests: `ts-host/test/escalate.test.mjs` (mock driver: floors, risk floor overrides argmax, nl policy invoked with
  the right facts, shadow event).

## 6. R5: one optimization ladder behind `natlang profile` (3 days, no deps)
natlang has six optimization entry points (`compilations`/`specialize`, `tune`, `optimize`/`adapt`, `improve`,
`check --fusion`, law rewrites gated off). v1 added a seventh (`natlang compile`). v2 adds no orchestrator command.
It adds one report and one decision:
1. `natlang profile [DIR] [--json]`: per function and inline site from the call store: calls, p50 latency,
   tokens per call, share served by compiled cases, decision confidence p10/p50, escalation rate, failure classes.
2. **Recommendation** ("which optimization next") is policy: `optimization/policies.ts` gains `nextOptimization`
   (pluggable; crisp rules default; `nl` = `applications/specializer`-style `.nl` reading the profile). The report
   prints the recommended command (`natlang specialize ...`, `natlang tune ...`, add `readout: decision`,
   `natlang improve ...`).
3. Law rewrites get a `natlang.json` key: `"rewrites": { "measurements": PATH }` passes a `RewriteGate` built from
   that file (`compiler/project.ts:638`). Rules stay off until a measurement certifies them, as `rewrites.ts`
   already requires.
- Tests: `ts-host/test/profile.test.mjs` (synthetic call store → report → recommendation; rewrites key reaches the
  gate).

## 7. Acceptance for WS2
- **Cache-friendliness.** `prompt-prefix-stability.test.mjs` passes: system message and tool block are byte-identical
  across call shapes on the ts-host renderer that feeds all four engines. R0's measurement, re-run after R1, shows
  the shared prefix per call is no shorter than before R1 on the same records. Per-call variable text starts only
  after the call's instructions.
- R1 is the default on main, with MODEL_FACING_CHANGES entries and the opt-out tested.
- R3 passes the parity guards on all four engines, and ENGINE_PARITY.md records it.
- `natlang profile` reports correctly on `examples/specialization` and `applications/helpdesk` fixtures.
