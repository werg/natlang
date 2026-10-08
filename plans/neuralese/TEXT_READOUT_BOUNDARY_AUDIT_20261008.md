# Neuralese text readout boundary audit — 2026-10-08

## Change

`TEXT_NEURALESE_EMULATION_PROMPT` revision 9 now describes the shared compiler contract: JavaScript text-coercion contexts read a `Neuralese<T>` operand at its declared type, then continue with native JavaScript conversion. The prompt names `String`, untagged template interpolation, string concatenation, `Error(message)`, `JSON.stringify(value)`, and supported native text methods. It also says this does not enable opaque field inspection or turn console display into readout. This repairs a prompt/code mismatch: the prior prompt said no general text conversion applied to other `Neuralese<T>` types, while the compiler already lowers explicit `String`, template, concat, `Error`, JSON, and declared text-method contexts through typed readout (`ts-host/src/compiler/neuralese.ts`, especially lines 360–415).

This wording change is a prospective ergonomics correction. The examined traces do not establish that the previous sentence caused a conversion failure.

## Actual trace evidence

- V25 `TREE-711`, child invocation `task-1-hgloot/18`, trace `runs/neuralese-semantic-iterate-reducers-v25-20261008-v1/generation-review-v1/luna/campaign-v1/slot-02/jobs/000001-1410734eae6fb02d.trace.jsonl`, seq 11: the `natlang.read` call has signature `(v: Neuralese<string>) => string` and returns `"[object Object]"` with no runtime diagnostic. This is a wrong learned readout value, not a JavaScript coercion or compiler rejection. The operator already requests an exact ordinary value of the declared type (`ts-host/src/neuralese/combinators.ts`, `COMBINATORS.read`). A post-hoc string filter would also reject a legitimate payload whose exact text is `[object Object]`; no recovery or normalization was added.
- V24 BB-541, child `task-1-3b0jyg/7`, trace `runs/neuralese-semantic-iterate-reducers-v24-20261008-v3/generation-review-v1/luna/campaign-v1/slot-04/jobs/000007-d137d6b7f202f825.trace.jsonl`: the first child attempt supplied a record where its declared callable expected a string, and the attempted result included `[object Object]`. This is a call-shape/type mismatch, not a safe stringification opportunity; the two-type-argument `nl.with<C,T>` capture/result form was implemented separately.
- V24/V25/V26 root traces also show `[object Object]` in console inspection of ordinary record locals. Console inspection is not a JavaScript text-coercion position; it remains outside typed readout.
- V24 TRN-520 has six direct `return_result` outputs whose string values contain one to three textual `[[Neuralese text block ...]]` wrappers and also add register, conditions, or authority content. These are authored strings containing transport-looking text, not soft block values. The existing authenticated `neuralese_code` sidecar adapter only validates an eval-code sidecar (`ts-host/src/compiler/neuralese-conversion.ts` and `training/neuralese/natlang_neuralese/data/text_corpus.py`); none of these `return_result` actions has such a sidecar. They remain in the candidate proposal's adapter-review backlog. No marker-looking string parser or segment extraction was added.

## Deliberate boundaries

- `String(Neuralese<Record>)` reads the declared record and then follows native JavaScript string conversion. It does not silently become JSON serialization.
- `JSON.stringify` remains a separate explicit native operation; only its actual typed-readout operand is materialized.
- Console display, arbitrary marker-looking strings, and wrapped `return_result` strings do not create typed reads or semantic writer evidence.
- The single observed bad `natlang.read` string has no detectable representation error: the model returned a string of the declared type. Improving it requires readout-quality evidence or a future qualified retry design, not a sanitizer that guesses content.

## Validation

Built TypeScript and ran `node --test test/text-neuralese-emulation.test.mjs` in isolated worktree `/tmp/natlang-vreadout-audit`; 10/10 text-emulation tests and 30/30 Neuralese compiler/runtime tests passed. No canonical `ts-host/dist` was rebuilt or changed. The tests cover the revised prompt revision and its scope alongside the existing typed-conversion compiler/emulation cases.
