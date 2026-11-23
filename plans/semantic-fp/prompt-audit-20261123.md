# Prompt audit 2026-11-23: cache map, stable prefix, sentence pass (B2; WS2 R0 + R1)

**Parent:** [02_RUNTIME_PROMPT_AND_COMPILATION.md](02_RUNTIME_PROMPT_AND_COMPILATION.md) §2.2-2.3,
[09_FIRST_BATCH.md](09_FIRST_BATCH.md) B2. Opt-out: `NATLANG_PROMPT_REVISION=2026-11-22` (texts and tool order before
this pass, byte for byte; `ts-host/src/native/prompt-2026-11-22.ts`). Records: `plans/MODEL_FACING_CHANGES.md`
2026-11-23 entries; DECISIONS 2026-11-23 "B2".

## 1. R0: the shared-prefix map

`scripts/neuralese_data/prompt_cache_map.py` (LFM3 tokens, the serving engine's own chat rendering; exact actor ids for
the collection). For each call: the longest token prefix shared with any earlier call of its episode and of its window
(the source in record order), and the region where the first differing token falls (system text, tool block, opening
message, later history, or none: the prompt is a prefix of an earlier one). "Recomputed" is prompt tokens minus that
shared prefix: what a cache that keeps every earlier prompt still has to prefill, and what the learner's prefix tree
computes per row beyond the shared part. Output: `ops/semantic-fp/prompt-cache/2026-11-23.json` (before) and
`2026-11-23-after.json` (the same records rewritten to this checkout's system prompt and tool list by
`ts-host/scripts/rewrite-recorded-prompt.mjs`).

Sources: the sft-v9 DGX held run (`ops/ei/sft-v9/held/start`, 5,376 calls), 40 groups of the sft-v9 collection
(`ops/ei/sft-v9/collect`, exact actor prompt ids), both VibeStudio live seeds, and two `examples/triage` runs
(`ts-host/scripts/capture-triage-requests.mjs`, scripted replies, real runtime prompts).

| Source | calls | prompt tok/call before -> after | system tok before -> after | recomputed tok per invocation opening, episode scope, before -> after | same, window scope | in-episode first differing tool of openings, before | after |
|---|---|---|---|---|---|---|---|
| sft-v9 collection (40 groups) | 2187 | 10895 -> 10271 | 3467 -> 2882 | 1985 -> 1773 | 198 -> 191 | {'read_code': 27, 'return_result': 25, 'compact_history': 20, 'edit_code': 5} | {'return_result': 77} |
| sft-v9 DGX held run | 5376 | 11786 -> 11171 | 3417 -> 2841 | 2447 -> 2209 | 409 -> 402 | {'compact_history': 91, 'return_result': 61, 'read_code': 26} | {'return_result': 178} |
| VibeStudio seed-v1 | 37 | 9595 -> 8967 | 3676 -> 3048 | 3234 -> 3186 | 3234 -> 3186 | {'return_result': 5} | {'return_result': 5} |
| VibeStudio seed-v2 | 58 | 9318 -> 8690 | 3676 -> 3048 | 3209 -> 3154 | 3000 -> 2972 | {'return_result': 11} | {'return_result': 11} |
| examples/triage x2 | 32 | 4944 -> 1411 | 3667 -> 872 | 1850 -> 1429 | 939 -> 753 | {'return_result': 4} | {'return_result': 1} |

"Recomputed" counts what a cache holding every earlier prompt of the scope still prefills. Episode scope is what a
per-slot cache (managed llama-server, in-page wllama) and a per-episode learner batch see; window scope is a large
radix cache (SGLang) holding every earlier call. The triage row is not a like-for-like pair: it was captured again
under this checkout, where the finite-result calls (classify, is_urgent) now run as decision readouts (16 of 22 calls,
no tools, their own system prompt), so its means mix two prompt families. The other rows are the same records.

What the map shows:
- **The system message never broke the prefix.** One system text per tool-loop call in every source (decision
  readouts have their own). An opening whose first difference is in the system text is an episode's first call, or a
  tool-loop call whose only earlier calls in the episode were decision readouts.
- **With a large radix cache the window already shared almost everything** (window hit rate 0.91-0.92 of all prompt
  tokens in the collection and the held run, before and after): every tool-list variant was cached after its first
  use, so the shape breaks cost little there. They cost in episode scope, which is what a slot cache and a batch of one
  episode's rows see.
- **The tool block did break it, in two places.** Calls of different shapes parted right after `read_page` (the file
  tools of a folder reducer were spliced in second) or after `read_code` (`edit_code`/`diff_code` for calls with code of
  their own), so the rest of the tool block (`read_code`/`compact_history`, about 300 tokens) was a separate prefix per
  shape. `return_result`, typed by the call's declared result, is per call by design (2026-11-12), so the first
  differing tool of calls of one shape is `return_result`.
- **Fix (open question 1):** the conditional `edit_code`/`diff_code` (added on 2026-11-17 in 1cb19e291 without a
  MODEL_FACING_CHANGES entry, against the 2026-11-12 "one constant opening" decision) do break the byte-stable tool
  prefix. They are not offered to every call again (a call without code of its own could only be refused, and the
  2026-11-13 entry removed them for that reason); instead the tool list now runs stable to variable: `eval`,
  `read_page`, `read_code`, `compact_history` for every call, then `escalate` (per run), `edit_code`/`diff_code`, the
  file tools, and `return_result` last (`agent.ts` `orderedTools`, `STABLE_TOOLS`). Every call of every shape shares
  the system message and those four tools; a variant costs only its own tail once per cache lifetime.
  `ts-host/test/prompt-prefix-stability.test.mjs` pins it for plain, own-code, folder (read/write, each file-tool
  surface), depth-limit, Neuralese and finite results and a real triage run (decision readouts included).
- After the pass the stable core is shorter (sentences removed), so the shared prefix in absolute tokens can be
  shorter while the recomputed tokens per call fall: the "after" column compares recomputed tokens, the quantity
  every engine and the learner pay.

## 2. Open question 2: SFT rendered own successes under the replaying runtime's prompt

`sft_sources build` rendered every replay-verified success from the *replay's* record
(`candidate_turns` -> `replayed_turns`), i.e. under the system prompt, tool list and openings of the checkout that ran
the replay. In `/home/werg/data/ei-sources` (the expert-iteration WORK) 5,833 of 10,540 replayed own successes (55%)
had a system prompt that differed from the one their calls were sampled with; the diff of 400 of them is the
1cb19e291 answer protocol (sampled under "reply with the declared string as plain text ... A reply without tools ends
your turn and is taken as your answer", trained under "A reply without a tool call is not an answer"). Fixed: own
candidates (`rl-own`, `jolt-guided`) render their recorded turns (`scripts/neuralese_data/recorded_turns.py`,
backfilled from WORK's extractions, written by `replay` for new episodes); the replay stays the verification.
`--own-render replayed` opts out. Teacher episodes keep the replayed rendering. Tests: `tests/neuralese/test_recorded_turns.py`.

## 3. R1: every sentence

Classes: **W** wrong or contradicting the runtime; **M** missing built-in; **S** task-specific scar tissue; **N** wrong
form spelled out; **A** anti-delegation; **E** copyable domain example; **D** duplicate; **C** capability text for some
calls; **K** correct and needed. Action: keep, delete, rewrite (new text shown).

### SEMANTIC_RESULT_PROMPT (system prompt, first paragraph)
| Sentence | Class | Action |
|---|---|---|
| Answer with the return_result tool: call it with status "success" and a value of the declared type (for a string result, the text itself). | K | keep |
| That value is the function's return value. | K | keep |
| Return only that value, without a completion note, explanation, Markdown wrapper, or extra fields unless the declared result calls for them. | K | keep |
| A reply without a tool call is not an answer. | K | keep (the TOOLS copy goes) |
| When the declared result is a string and the task asks you to save JSON and return it, return exactly the JSON text saved in the file as that string. | S | delete |
| Send it as the value of an actual return_result tool call; do not print a pseudo invocation, status envelope, explanation, or code fence instead. | N, D | delete (the first sentence says the positive form; a prose reply is answered by the runtime) |
| For a boolean proposition, true means that the stated proposition applies ...; keep any negation in the question. | K | keep |
| A declarative proposition in a boolean task is a claim to evaluate against the inputs, not a fact to assume. | K | keep |
| Use eval for exact computation and effects; eval {code, finish:true} computes and completes ... in one action. | K | keep |
| A successful finish:true eval ends this call immediately; include every required predicate ... and do not send another conflicting final action. | D, N | delete (the eval tool's description says it) |
| Never present code as though it had run. | D | delete ("Code runs only in eval.") |
| To inspect evidence in eval, use console.log(value). | K | keep |
| A top-level return proposes the function result; reserve it for a value of the declared result type. | K | keep |
| Neuralese<string> parameter and result paragraph (piece `neuralese/string-boundaries`) | C | keep verbatim: shared by every call, so moving it saves nothing after caching; no plain call in the R0 records produced Neuralese markers |

### SEMANTIC_CALL_COMPLETION_PROMPT
| Sentence | Class | Action |
|---|---|---|
| For a semantic judgment over one supplied item, inspect that item and make the judgment in this call. | A | replaced |
| A child receives the responsibility to answer its assigned question. | K | replaced |
| Do not delegate the same question over the same unchanged input to another child. | N | replaced (the runtime refuses it: `SAME_QUESTION_REFUSAL`, kernel.ts) |
| This applies to inline nl children: when the child's inputs ... are enough ..., return that declared typed answer directly. | A | replaced |
| Nest another nl call only for a genuinely distinct subproblem ..., not a near-identical re-ask. | N | replaced |
| When the evidence is already available, return the declared value directly. | A | replaced |
| New: "Give a child call a distinct sub-question and the inputs it needs; answer the question you were given here." | | |

### TOOLS_PROMPT
| Sentence | Class | Action |
|---|---|---|
| You are running one call of a natural-language function. Execute the function to completion, ...; return the actual result. | K | keep |
| Use eval for exact computation, transforming data, calling available functions and services, or working with files. Code runs only in eval. | K | keep |
| A reply without a tool call ends your turn without answering. | D | delete |
| Your tools are eval, return_result and the others listed. Invoke tools such as return_result directly as tool calls. read_code is also available as ... eval function ...; the tool call itself takes { name }. Other tools are not functions in eval. | K | keep |
| Everything else ... lives in the eval scope ...; for example const firstPage = facts.page(1), and read what comes back. | K | keep |
| Call arguments are recursively read-only. To edit an argument tree or array, work on a mutable copy (...) and return that copy. | K | keep (the one deep-copy sentence; GENERATION_GUIDANCE's now refers to it) |
| When the instruction asks for a decision, extraction, or revised notes, return that value. | S, D | delete |
| Use input and captured files as evidence; save or edit a file only when this call asks for that file change. | K | keep |
| REPL scope, first eval, cut-off names, transcript, compact_history, await (six sentences) | K | keep |
| Eval code is ordinary TypeScript with a few limits: ...; while and do are refused, functions may not call themselves recursively, ... | K | keep (true until WS1 L7 lands; then "only on a smaller argument") |
| Eval code can also hand a judgement ... to a new natural-language function: nl`instructions` creates one inline, ... | K | keep |
| Use inline nl lambdas ...: const policy = 'Refunds are allowed within 30 days of delivery.'; const verdicts = ... nl<boolean>`Decide whether item meets policy.` ... | E | rewrite: `const verdicts = await Promise.all(<items>.map(item => nl<boolean>`Decide whether item meets <criterion>.`(item))); const kept = <items>.filter((item, i) => verdicts[i] === true), where <angle brackets> stand for this call's own names and criterion.` |
| Use Boolean child results for eligibility checks; objects and strings are not Boolean verdicts. | K | keep |
| nl lambdas participate in type inference ... const risk: 'low' \| 'high' = await nl`Rate the risk in note.`(note). | K | keep |
| When nothing says it ... nl<{ level: number; reason: string }>`Assess the risk in note.`(note). | K | keep |
| Variables in scope that its instructions mention by exact name are visible to it. | K | keep |
| To list shared context explicitly, use nl.with<boolean>({ policy })`Decide whether item meets policy.`(item). | E | rewrite: `nl.with<boolean>({ <criterion> })`Decide whether item meets <criterion>.`(item)` |
| nl.with<ResultType> ...; the summarize/rubric example | K | keep |
| The Neuralese revise/decisionRule example (three sentences) | C | keep (shared prefix; see Neuralese above) |
| Keep input FileHandles in the arguments; return the requested result, not the input/context wrapper. | K | keep |
| The listed values are snapshots ... Pass changing iteration state as an argument ... | K | keep |
| When instructions define ordered repair passes, apply only the current pass's correction ... | S | delete |
| Return the declared draft value itself, rather than wrapping it in another object. | S, D | delete |
| You can keep the instructions focused on the judgment and refer to the bound context by name ... | K | keep |
| If you get a question about meaning, don't try to solve it via code, keyword or regular-expression matching; Solve it by looking at the data ... | N | rewrite: "Answer questions about meaning by reading the data, or by mapping an nl function over it when it is large." |
| Prefer a function you can already call that makes the judgment over creating a new one. | K | keep |
| When delegating, preserve the parent's criterion and relevant context. Do not add exclusions or require stronger evidence than the parent asks for; a question can express the same intent as a statement. | N | rewrite: "When delegating, pass the parent's criterion as it stands, with its relevant context; a question can express the same intent as a statement." |
| (new) decide | M | add: "For a function with a finite result (a boolean or a union of literals), decide(fn, ...args) runs it and returns { value, probabilities, confidence }: use it when how sure the answer is matters to what you do next." |
| iterateOn: operator, bound before .until, known list of edits | K | keep |
| For an available callable stepFn, stepFn.iterateOn(...) already supplies stepFn; never pass the callable again as the initial state. | N | rewrite: "... supplies stepFn itself, so its first argument is the initial state." |
| Free form; NL method with the deathStar example; latest state; build each revision from the step argument; state type; stopping check runs first; completion flags | K | keep |
| For pagination, start with more: true, accumulate each response page, ... | S | delete (also in the depth-limit paragraph) |
| return_result statuses; a final return stages the value | K | keep |
| When the declared result is exactly Neuralese<string>, return ordinary text ...; If no writer is configured, the runtime reports that capability error. | D | delete (the Neuralese paragraph says it; the error says the rest) |
| For an ordinary string result, marker-looking characters are literal ...; For other Neuralese<T> positions, pass the existing typed value or one exact marker ... | C | keep |
| Output or values too long to show are cut off, marked <<cut off: …>> ... which read_page call shows the next part. | K | keep |
| The marker is not code; never copy a cut-off value, use the variable that holds it. | N | rewrite: "The marker is not code: use the variable that holds the value." |
| When exact text is supplied in structured data, parse and use its string fields directly ... | K | keep |
| When separate files pair a question or criterion with a state, ... Do not merge facts across those states ... Bind each output key to its matching file ... | S | delete |
| For a conversation state, read the full exchange and reconcile earlier claims ...; a proposed remedy does not establish that it succeeded. | S | delete |
| Verify the resulting file against those fields. | S | delete |
| A find/replace request applies to the matching span ... Preserve text outside that span, including leading or trailing whitespace and final newlines. | S | delete |
| Missing-information paragraph ("Use the information that the instructions point to." ... "When further searching keeps finding nothing new, stop ...") | K | keep |
| When an unchanged candidate, child update, or action has already been rejected, use the observed reason ... | K | keep |
| Use the argument names in this call's displayed signature ...; mentioning item or file in an instruction does not create a variable ... | K | keep |
| A delegated call must allow the same successful negative or missing-information result as the parent task. | K | keep |
| Partial checks, child inheritance, entity/time/definitions, output contract (four sentences) | K | keep |
| For a state revision, pass the full current carried value ..., have it return the declared child type, and preserve fields outside the current pass. | S (tail) | rewrite: "... as its argument and have it return the declared child type." |
| For prose notes, preserve supported facts rather than exact wording, unless the task requires verbatim copying. | S | delete |
| In iterateOn, build the next state from the step's current state and returned value, never from a captured outer initial draft. | D | delete (the iterateOn paragraph says it) |
| An isolated quote can lose the scope established by its containing record or folder. | S | delete |
| Within one complete record, carry its stated entity or group scope across sentences ... | S | delete |
| Keep facts from different records separate unless the task asks for a join. Evaluate each assigned condition independently ... Combine those checks in the parent ... | K | keep |
| When a search or helper returns no matches, re-read the original question ... Rephrasing a query with the same unsupported premise adds no evidence. | K | keep |
| In TypeScript, a map callback that uses await must be async: ... | K | keep |
| Be brief. ... | K | keep |
| For exact array work, calculate directly in eval ...; the sorted-count example; bounded iterateOn; Promise.all aggregation | K | keep |
| Use the actual input and service names listed in this call; do not redeclare or shadow those bindings or invent function names. | N | rewrite: "Use the actual input, function and service names listed in this call." (`invalid-binding` teaches the rest) |
| Complete a typed child answer concisely, and use the eval tool with finish:true only when its result and required effects are ready. | K | keep |

### GENERATION_GUIDANCE (and its depth-limit form)
| Sentence | Class | Action |
|---|---|---|
| For a large input, use eval to select the fields or subtree relevant ...; read more only when needed. | K | keep |
| When dividing work, pass each helper the actual item or extracted slice ... | K | keep |
| For a tree edit, deep-copy the input, change only the requested leaf ..., then inspect that path. Once the path is known, verify that path directly; a recursive tree walker or an open-ended loop is unnecessary. | D, S | rewrite: "For a tree edit, change only the requested leaf of the mutable copy at its existing path, preserve wrappers and siblings, then inspect that path." |
| For exact numerical answers, identify the requested operands and their units ... | K | keep |
| Before returning, check that each value answers the requested predicate ... | K | keep |

### FUNCTION_TOOLS_PROMPT (call note of calls with code of their own)
Every sentence **K** or **C** (read_code/edit_code, reducers' apply/propose/accept/select, folder iterateOn with a
measure): kept verbatim. It is already a per-call note after the instructions, and only calls that can use it get it.

### directoryReducerPrompt (call note of folder reducers)
| Sentence | Class | Action |
|---|---|---|
| Declaration, paths, the file-tool list, folder/file handle API, moves, Node modules unavailable, the JSON-report finish example, literal text | K | keep |
| inline nl per-file judgments with the inbox example | K | keep |
| When an input file supplies an exact edit request, parse it in code and pass its strings directly to the edit operation. | K | keep |
| Do not retype, trim, or extend the supplied match or replacement. | N | delete (the sentence before says the positive form) |
| For change-request.json: const change = ...; If the task asks for the target path, finish with that path string; "done" is appropriate only when the task asks for it. | S | delete |
| Subfolder reducer, child root, large file handles | K | keep |
| For a table joined to message files, parse the original table ...; use the original amounts rather than retyping a second amount table. | S | delete |

### Not changed, and why
- **Neuralese sentences in the stable core** (~1.5k characters): every call shares them, so moving them to a call note
  saves nothing after caching and would make every Neuralese-using call pay them; R0's records show no plain call
  misled by them.
- **`prompt-example` refusal kept**, narrowed: the system prompt's examples are schematic now, but `read_code`'s `nl`
  documentation (`native/runtime.ts` BUILT_IN_DOCS.nl) still shows `Decide whether item meets policy.` (09_FIRST_BATCH
  B2 assumed the example would no longer exist anywhere). `PROMPT_EXAMPLE_TEXTS` is that one text; the refusal names
  read_code's documentation as its source.
- **Prompt soft forms:** no deployed prompt-state build holds a changed piece (the operators' builds hold `view@...`
  only); `promptStateIds` refuses a state primed from another text (`system-prompt-stale`), so a stale soft form can
  never stand for the new text. A new interpreter bank is built from the new text with
  `ts-host/scripts/neuralese-system-prompt-bank.mjs --pieces interpreter,interpreter/depth-limit,generation-guidance,...`
  when one is wanted.
