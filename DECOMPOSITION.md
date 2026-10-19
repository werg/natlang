# Native structured completion to typed result

## Natlang-native policy

The existing typed function contract decides whether a completion is an ordinary value or a Neuralese value. A returned state is kept as a reference only when that declared result type accepts a single state reference. Otherwise its authenticated stored text is the source representation for the declared ordinary result; the existing runtime performs the normal exact type check and JSON parsing. Mixed text/state content is read in order by using each state's stored text. Missing state text or artifact identity fails closed.

## Mechanism

1. The model server continues to report actual text and native-state content parts and archives every reported state before the agent normalizes the turn.
2. The shared TS turn decoder receives the declared return type and the native-state store already owned by that call.
3. For a value boundary, it preserves native-state parts when the declared type admits Neuralese (or `unknown`), including through `refined`/`untrusted` wrappers and list alternatives inside unions; list alternatives become the existing typed sentinels. Otherwise it materializes content in order, resolving native-state parts to their authenticated `metadata.text` and preserving ordinary text runs. Textual producer spans still use the existing `written` accumulator and graph recording.
4. The existing `return_result`/`coerceReturn` path validates the resulting declared type. No type checks, state identity checks, or graph receipts are bypassed. The inferred assistant completion is passed through the existing typed `return_result` path; explicit tool-call values keep their existing transport handling.

## Exact code

- Extend `decodeTurnValue` in `ts-host/src/native/neuralese.ts` to accept an optional authenticated native-state text resolver and use the existing type environment to choose state-reference preservation versus text materialization.
- Pass the call's existing `NeuraleseStore.getNativeState` resolver from `NativeAgent.decodeNeuralese` only for the inferred assistant result; do not reinterpret explicit tool arguments or explicit `return_result` values.
- Add tests at the actual `NativeToolAgent` boundary for ordinary string/number/object results, `unknown`, Neuralese lists/records, a genuine textual producer span, and mixed text/state order. A missing or unavailable native artifact remains an error.
- Update authoring/integration recovery guidance only for the concrete completion conversion semantics. Prompt text and model-visible wording remain unchanged; measure the new behavior on the next authorized teacher window and record measurement debt until then.

## Shared-state writer binding

### Natlang-native policy

The declared native model service is the producer: it receives source text and creates authentic shared positional/hybrid state under its own tokenizer, model, and producer markers. The host configures that service and its archive; it must not synthesize a replacement state from text or projected values.

### Mechanism

1. The caller supplies endpoint, headers, and native-capable store.
2. `nativeStateRuntimeOptions` binds the server's declared dialect, reader identity, and existing `native.text-write` endpoint. Required writers fail during setup when that capability is absent; read-only callers may request the explicitly optional writer mode.
3. Collector, combinator initialization, and direct runtime harnesses consume this single configuration instead of assembling partial store-only settings.
4. The existing runtime write path and endpoint validate exact source text, token span, producer markers, checkpoint/tokenizer identity, and archive the actual state artifact.

The public 48-session probe passed only `{store}` and therefore never configured a producer port. It found a harness wiring omission; the reference server already implements the real native producer endpoint. Corrected public sessions must use `nativeStateRuntimeOptions` and report actual returned state provenance.


### Complete-gradient reference accumulator placement (2026-10-20)

The supervisor chooses the useful training backend. The exact verifier compares every declared gradient against explicit cached execution. Its cached backward retains interleaved expert cotangents until their shared leaf completes, so GPU accumulation can exhaust device memory independently of the candidate training path.

1. Keep the candidate scope on the model-selected backend.
2. Collect the explicit cached reference with the existing CPU-file FP32 accumulator, preserving contribution order and the final parameter-dtype cast.
3. Record both observed accumulator backends, compare all parameter and input gradients at the existing gates, and restore prior gradients and RNG.

These are crisp verifier/storage mechanisms; no data, trainable coverage, numerical gates or production optimizer behavior changes.
