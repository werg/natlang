# Error codes

Every error code that [SPEC.md](SPEC.md), the [extensions](ext/neuralese.md), the Neuralese documents and the skills
name, with the rule it enforces, when it fires and the file that raises it: a `ts-host/src` file for the runtime, or a
file of the reference model server (`training/neuralese/natlang_neuralese/serve/`, [NEURALESE_PORT.md](NEURALESE_PORT.md))
for the codes it answers. Each code was checked with `grep` against those sources; the mismatches are listed at the end.

**When it fires.**

- **Plan time**: reported by `natlang check` / `natlang build` before any model runs (the compiler plans every `nl`
  expression), or when a type is read.
- **Load/bind time**: when a context, file or `.nl` source is loaded, bound or saved.
- **Eval time**: when the model's eval code is checked or committed, before or while it runs. The model sees the
  rejection and repairs it.
- **Call time**: while a call or an `iterateOn` loop runs, or when a value crosses a refined slot.

## Codes

| Code | Rule | Fires | File |
| --- | --- | --- | --- |
| `nl-ambiguous-signature` | Conflicting evidence about an `nl` expression's parameters or result ([SPEC](SPEC.md#natural-language-functions)) | plan time | `compiler/inline.ts` |
| `nl-unknown-parameter` | The type found for a parameter cannot be used as a signature type; annotate it | plan time | `compiler/inline.ts` |
| `nl-unknown-return` | The return type of the `nl` expression cannot be determined from its target; annotate it or write `nl<T>` | plan time | `compiler/inline.ts` |
| `nl-sync-callback` | An `nl` function (async) put in a callback slot that expects a synchronous result | plan time | `compiler/inline.ts` |
| `nl-sequential-loop` | Warning: a loop of independent `nl` calls awaited one at a time; use `Promise.all` | plan time (`natlang check`, warning) | `compiler/sequential-loops.ts` |
| `type-recursive-function` | A type alias may not mention itself in a function parameter or result position ([SPEC](SPEC.md#iteration-and-termination)) | plan time | `compiler/neuralese.ts`, `native/types.ts` |
| `undeclared-field` | Eval reads a field the declared type lacks ([SPEC](SPEC.md#types)) | eval time (checked before the eval runs) | `compiler/eval-check.ts` |
| `invalid-binding` | An eval binding that would shadow or read itself before it exists (`const plan = plan(...)`) | eval time (before the eval runs) | `scope-compiler.ts` |
| `capture-conflict` | A `let` capture changed under the eval; the write-back is version-checked ([SPEC](SPEC.md#captures)) | eval time (write-back) | `native/runtime.ts` |
| `context-interface-mismatch` | A rebound or replaced context does not satisfy the function's context interface ([SPEC](SPEC.md#contexts)) | load/bind time | `runtime/contexts.ts` |
| `context-new-executable` | Rebinding may not add an executable node | load/bind time | `runtime/contexts.ts` |
| `iteration-unbounded` | An `iterateOn` with a TypeScript predicate needs a measure or step limit ([SPEC](SPEC.md#iteration-and-termination)) | call time, before the first step (`IterationLimitError`) | `runtime/iterate.ts` |
| `iteration-measure-exhausted` | The measure reached 0 with `until` still false (`IterationLimitError`) | call time | `runtime/iterate.ts` |
| `refinement-unsatisfied` | The predicate was judged false ([refinements](ext/refinements.md)) | call time | `native/refinement.ts` |
| `refinement-undecided` | The probability fell in the band under the `reject` policy, or no judge was available | call time | `native/refinement.ts` |
| `refinement-predicate-invalid` | `P` in `Is<T, P>` is empty or not a string literal | plan time (type read); call time for `assume` | `native/types.ts`, `native/refinement.ts`, `runtime/surface.ts` |
| `untrusted-instruction` | An `Untrusted<T>` value interpolated into an inline `nl` template | plan time | `compiler/inline.ts` |
| `neuralese-untyped-literal` | A Neuralese literal or soft function with no contextual type ([neuralese](ext/neuralese.md)) | plan time | `compiler/neuralese.ts`, `compiler/inline.ts`, `scope-compiler.ts` |
| `neuralese-opaque-access` | Host code accessing fields or indices of, or computing with, a soft value | plan time | `compiler/neuralese.ts` |
| `neuralese-condition` | A soft value used as a condition | plan time | `compiler/neuralese.ts` |
| `neuralese-readout-sync` | Implicit string conversion of a soft value in a synchronous function or callback | plan time | `compiler/neuralese.ts` |
| `neuralese-nested` | `Neuralese<Neuralese<T>>` | plan time (type read) | `compiler/neuralese.ts`, `native/types.ts` |
| `neuralese-dialect-mismatch` | A `.nz` entry's block dialect differs from its declared type; a block coerced to a `Neuralese` type (argument, result, `let`, `read`) was stored in another dialect than the type names (`DefaultDialect`: the runtime's reader dialect); text at a slot of another dialect than the port's; a runtime whose declared dialect differs from its port's | load time, call time, runtime creation | `native/nz-file.ts`, `native/values.ts`, `native/runtime.ts`, `neuralese/combinators.ts`, `native/neuralese.ts` |
| `neuralese-block-integrity` | A `.nz` block does not hash to its ID | load time | `native/nz-file.ts` |
| `neuralese-live-capture-save` | A Neuralese function with live captures cannot be saved to a file ([SPEC](SPEC.md#captures)) | save time | `native/nz-file.ts`, `runtime/contexts.ts` |
| `neuralese-readout-unavailable` | Implicit string conversion with no configured read body | call time | `neuralese/combinators.ts` |
| `neuralese-unsupported-backend` | A call needs Neuralese on a backend without support; there is no text fallback; a generic result's Neuralese instance on a runtime without a reader dialect (the message names the call site) | call time | `native/neuralese.ts`, `runtime/kernel.ts` |
| `neuralese-crisp-result` | A natlang function with a crisp result called where a `Neuralese` value is expected; the message shows the generic result that makes it write ([neuralese](ext/neuralese.md#generic-results)) | plan time | `compiler/neuralese.ts` |
| `neuralese-native-consumer-unavailable` | A native state from another native or semantic dialect reaches a `DefaultDialect` slot, and the reader declares neither text re-encoding nor an adapter that can consume it ([neuralese](ext/neuralese.md)) | call time | `native/values.ts` |
| `neuralese-native-text-write-unavailable` | Writing text as a `Neuralese<string>` needs a server that lists `native.text-write` | runtime creation, call time | `model/native-state-client.ts`, `neuralese/combinators.ts` |
| `neuralese-native-empty-unavailable` | `empty()` needs the loaded native library's archived zero-token state | call time | `neuralese/combinators.ts` |
| `neuralese-native-archive-unavailable` | A native operation needs a store that keeps native states and tensor artifacts | call time | `model/native-state-client.ts`, `native/neuralese-store.ts`, `neuralese/learning.ts` and other native clients |
| `neuralese-native-adapters-unavailable` | A call binds weight adapters, but its transport or model server does not apply them (`native.adapters`) | call time | `model/chat-completion.ts`, `model/neuralese-server.ts` |
| `neuralese-native-tensor-as-state` | A named tensor artifact (`nnt1_…`) used as communication state | call time | `native/neuralese.ts` |
| `neuralese-native-coefficient-policy` | A call's actor coefficients differ from the declared actor policy and its coefficient bundle | call time | `model/native-coefficient-policy.ts` |
| `neuralese-native-source-integrity` | A native write's archived state or source text differs from what was supplied | call time | `model/native-state-client.ts` |
| `neuralese-native-protocol-integrity` | A server's marker-failure observation lacks its typed failed turn or receipt | call time | `native/native-protocol.ts`, `native/agent.ts` |
| `neuralese-native-gradient-binding` | A returned gradient does not belong to its exact target root and actor | call time | `neuralese/native-learning-client.ts` |
| `neuralese-native-update-binding` | An optimizer result does not belong to the published update or checkpoint | call time | `neuralese/native-learning-client.ts` |
| `learning-native-target` | A gradient or optimizer step without archived native targets (`nzs1_` states, `nnt1_` coefficients) ([learning](ext/learning.md)) | call time | `neuralese/learning.ts` |
| `learning-privilege-leak` | `conditionedDistill`: the student's view contains the privileged text | call time | `neuralese/learning.ts` |
| `learning-teacher-unconditioned` | `conditionedDistill`: the teacher saw exactly what the student saw | call time | `neuralese/learning.ts` |
| `neuralese-file` | A `.nz` file in the project that cannot be loaded (the message carries the loader's reason) | plan time | `compiler/project.ts` |
| `neuralese-file-format` | A `.nz` file that is not a native `natlang.neuralese-file/1` file (a projected `blocks` header, overlapping or missing inventories) ([NEURALESE_FILES](NEURALESE_FILES.md)) | load time | `native/nz-file.ts` |
| `neuralese-file-type` | A `.nz` export whose value does not have its declared type | load time | `native/nz-file.ts`, `runtime/contexts.ts` |
| `neuralese-ref-cycle` | A `.nz` export that refers to itself through `$ref` | load time | `native/nz-file.ts` |
| `neuralese-unknown-block` | A `.nz` file or browser archive that lacks a native artifact one of its values uses | load time, save time | `native/nz-file.ts`, `browser/neuralese-opfs-store.ts` |

## Model server codes

Answered by the reference server as `{"error": {"code", "message"}}` with the status shown
([NEURALESE_PORT.md](NEURALESE_PORT.md) "Conventions"). The runtime reports them as the call's error.

| Code | Rule | Status | File (`serve/`) |
| --- | --- | --- | --- |
| `bad-json` | The body is not a JSON object, or `collect`'s `referenced` is not a list of IDs | 400 | `http.py` |
| `bad-request` | A malformed chat field (`x_natlang_native_protocol_recovery` not boolean, a forced message without recording) | 400 | `http.py` |
| `bad-max-tokens` | `max_tokens` or `max_completion_tokens` is not a non-negative integer | 400 | `http.py` |
| `unsupported-request-field` | A retired projected-write field (`neuralese_temperature`, `neuralese_max_length`, `neuralese_length`, `neuralese_passes`, `x_natlang_forced`) | 400 | `http.py` |
| `prompt-rendering` | The messages do not fit the server's declared prompt rendering | 400 | `engine.py` |
| `empty-prompt` | The request renders no prompt | 400 | `engine.py` |
| `context-length` | The prompt, a template prefix or a view site does not fit the served context | 400 | `engine.py`, `http.py` |
| `tool-arguments` | Tool-call `arguments` given as JSON text that does not parse | 400 | `chat.py` |
| `forced-message` | Teacher-forced replay without structural recording, without its bound source receipt or teacher, or longer than the token allowance | 400 | `engine.py` |
| `native-rl-sampling-law` | `x_natlang_rl_capture` at temperature 0 without structural recording | 400 | `engine.py` |
| `neuralese-bad-part` | A `native_state` part with a malformed ID, `stop_gradient` or `source_occurrence` | 400 | `chat.py` |
| `neuralese-retired-part` | A content part of the retired `"neuralese"` type | 400 | `chat.py` |
| `neuralese-render` | The chat template dropped or duplicated a native-state reference | 400 | `chat.py` |
| `neuralese-template` | A malformed `neuralese_template` | 400 | `chat.py`, `engine.py` |
| `neuralese-guidance-template-conflict` | A forced template that guidance would reject | 400 | `native_guidance.py` |
| `neuralese-native-state-open` | A reply ended, or reached `max_tokens`, inside a producer span | 400 | `engine.py`, `native_protocol.py` |
| `neuralese-native-state-close` | A producer end marker with no open span | 400 | `engine.py`, `native_protocol.py` |
| `neuralese-native-state-nested` | A producer start marker inside an open span | 400 | `engine.py`, `native_protocol.py` |
| `neuralese-native-state-consume-marker` | The model drew a consume marker, which only the server inserts | 400 | `engine.py`, `native_protocol.py` |
| `neuralese-native-state-mode` | An invalid `same-base-adapter-state` declaration | 400 | `engine.py` |
| `neuralese-native-state-empty-text` | Text re-encoding of a state that has tokens but no text | 400 | `engine.py` |
| `neuralese-native-state-text` | `write-text` or `init-body` without usable text | 400 | `engine.py` |
| `neuralese-native-state-placeholder` | An `init-body` placeholder that does not occur exactly once, or as selected | 400 | `engine.py` |
| `neuralese-semantic-adapter-mismatch` | Semantic adaptation of a state failed against the server's adapter | 400 | `engine.py` |
| `neuralese-native-actor-changed` | The adapter binding changed during a captured request | 400 | `engine.py` |
| `neuralese-native-adapter-invalid` | Malformed or unknown actor adapters, or an adapter creation the server refuses | 400 | `http.py`, `engine.py` |
| `neuralese-decision` | A decision without options, with non-string options, or whose options render different prompts | 400 | `native_learning.py` |
| `neuralese-view`, `neuralese-view-source`, `neuralese-view-window` | A view write produced no state or no sampling receipt; a non-positive `window` | 400 | `http.py` |
| `neuralese-native-grad-schema`, `neuralese-native-grad-order`, `neuralese-native-grad-graph`, `neuralese-native-grad-objective`, `neuralese-native-grad-target` | A gradient request with the wrong schema, a derived or second-order graph request, a missing graph row, an unknown objective, or a bad target | 400 | `native_grad.py` |
| `neuralese-native-grad-invalid` | A gradient request that fails validation (graph, recorded-objective and derivative-program forms, decision objectives included) | 400 | `native_grad.py`, `http.py` |
| `neuralese-native-optim-schema`, `neuralese-native-optim-invalid` | An optimizer request with the wrong schema, or one that fails validation | 400 | `native_optim.py` |
| `neuralese-bad-native-state`, `neuralese-bad-native-artifact` | An uploaded body that does not decode, or that the store refuses (a failed derivation or gradient-origin check, a content collision) | 400 | `http.py` |
| `neuralese-id-mismatch` | An uploaded body hashes to another ID than its path names | 400 | `http.py` |
| `neuralese-unknown-native-state`, `neuralese-unknown-native-artifact` | The store lacks the state or artifact a request names | 400 in a request, 404 for a fetch or pin | `http.py`, `engine.py` |
| `not-found` | An unknown path | 404 | `http.py` |
| `method-not-allowed` | A method other than GET, PUT and POST | 405 | `http.py` |
| `internal` | An unexpected failure | 500 | `http.py` |

## Error classes

These are exceptions rather than codes. `NatlangCallError` carries `outcome` (`quiesced` or `failed`) and `detail`
([SPEC](SPEC.md#natural-language-functions)); `RefinementCallError` is a `NatlangCallError` with a refinement code;
`RefinementError` is thrown for a rejected argument, service result or `refine`; `NatlangRecursionError`
(call time) is a recursive call that is not on a smaller argument; `IterationDivergedError`, `IterationStepError`
and `IterationLimitError` (call time) carry `lastState` and the trajectory; `NzFileError` carries the
`neuralese-*` file codes above. All exist in `ts-host/src` (`runtime/kernel.ts`, `runtime/index.ts`,
`runtime/iterate.ts`, `native/nz-file.ts`).

## Mismatches found by the grep

1. `neuralese-dialect-mismatch`: resolved 2026-10-09 for the runtime (stored block dialects are checked at every
   coercion to a `Neuralese` type, DECISIONS.md "representation chosen by use"). The compiler still has no check of
   that name: it sees `DefaultDialect` as its own literal, not the configured dialect.
2. `TypeSyntaxError` and `EvaluationInfrastructureError` are named in the skills but carry no code.
3. Backticked kebab-case words in the docs that are not error codes and have no source in `ts-host/src` as errors:
   `value-not-printed`, `crisp-value`, `single-use`, `full-value-unavailable`, `producer-missing`, `dynamic-text`
   (training-data conversion reasons in `spec/NEURALESE_DATA.md`, found in `compiler/neuralese-conversion.ts`);
   `guidance-only` (an evaluation category in `evaluation/types.ts`); and the rewrite-rule names
   `map-identity`, `map-fusion`, `read-map`, `combine-identity`, `combine-reassociate`, `split-zip`
   ([NEURALESE_REWRITES.md](NEURALESE_REWRITES.md)).
4. Codes in the source that the docs do not name: `nl-not-called`, `nl-not-tag`, `nl-shadowed`, `nl-explicit-captures`,
   `nl-type-arguments`, `nl-parameter-collision`, `nl-unknown-name`, `nl-spread`, `nl-const-capture-write`, `reserved-property`, `duplicate-site`, `context-conflict`, `forbidden-control`,
   `forbidden-loop`, `forbidden-prototype-mutation`, `recursion`. They are outside this index until a document names
   them.
