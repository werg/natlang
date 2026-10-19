# Native structured completion to typed result

## Natlang-native policy

The existing typed function contract decides whether a completion is an ordinary value or a Neuralese value. A returned state is kept as a reference only when that declared result type accepts a single state reference. Otherwise its authenticated stored text is the source representation for the declared ordinary result; the existing runtime performs the normal exact type check and JSON parsing. Mixed text/state content is read in order by using each state's stored text. Missing state text or artifact identity fails closed.

## Mechanism

1. The model server continues to report actual text and native-state content parts and archives every reported state before the agent normalizes the turn.
2. The shared TS turn decoder receives the declared return type and the native-state store already owned by that call.
3. For a value boundary, it preserves a sole native-state part when the declared type admits Neuralese (or `unknown`); otherwise it materializes content in order, resolving native-state parts to their authenticated `metadata.text` and preserving ordinary text runs.
4. The existing `return_result`/`coerceReturn` path validates the resulting declared type. No type checks, state identity checks, or graph receipts are bypassed. The inferred assistant completion is passed through the existing typed `return_result` path; explicit tool-call values keep their existing transport handling.

## Exact code

- Extend `decodeTurnValue` in `ts-host/src/native/neuralese.ts` to accept an optional authenticated native-state text resolver and use the existing type environment to choose state-reference preservation versus text materialization.
- Pass the call's existing `NeuraleseStore.getNativeState` resolver from `NativeAgent.decodeNeuralese` only for the inferred assistant result; do not reinterpret explicit tool arguments or explicit `return_result` values.
- Add tests at the decoder and actual `NativeToolAgent` boundary for ordinary string/number/object results, a genuine Neuralese result, and mixed text/state order. A missing or unavailable native artifact remains an error.
- Update authoring/integration recovery guidance only for the concrete completion conversion semantics. Prompt text and model-visible wording remain unchanged; measure the new behavior on the next authorized teacher window and record measurement debt until then.
