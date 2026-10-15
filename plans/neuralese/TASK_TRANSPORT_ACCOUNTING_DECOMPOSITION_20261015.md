# Preserve interrupted model-request evidence in task accounting

Readiness execution accounting (2026-10-16): a cadence evaluation already
distinguishes fully accounted model failures from missing infrastructure
observations. Its `complete` field separately measures result exports. Native
readiness incorrectly required both, preventing known model incompletion or
token-ceiling failures from progressing even though task success is telemetry.

1. Produce a version 2 native execution receipt from the exact saved score
   bytes. Verify summary equality, selected-case count and unique identities,
   detailed outcome counts and total success count. Unknown/infrastructure
   outcomes, missing cases, historical scores and mismatched current identities
   remain disqualifying.
2. Define execution completion as all selected outcomes accounted. Preserve
   `result_export_complete` separately, unchanged from the source score, and
   retain all model failures and successes exactly. Never fabricate exports or
   silently reinterpret a version 1 receipt.
3. Reuse one receipt builder in native SFT and semantic training. Both raw
   readiness and shared-stage completion verify the bound score and version 2
   receipt. Existing version 1 receipts retain their original stricter contract.
4. Keep actual task performance visible and diagnose failures while training
   progresses. Neither this fix nor a successful execution receipt claims
   model quality, completed data exposure or numerical qualification.

Ledger validation: `native-accounted-execution-receipt-20261016` passed all
67 receipt/readiness/native SFT/shared-stage tests in 266.43 seconds;
`native-accounted-execution-final-20261016` passed 18 focused tests after
malformed-input guards; quick checks passed. Independent review accepted
both validator call sites and current-checkpoint binding. The receipt is
trusted evaluator evidence; score JSON alone does not cryptographically prove
which weights executed. Historical reports remain unchanged.

The existing task verifier must distinguish model failures from infrastructure interruptions. A real saved case finished its first8192-token reply, then its second HTTP request terminated after120.541seconds with TypeError: terminated; no result row was produced.

1. Existing queue accounting reads the exact index/program-bound error file; retain its error, path and SHA in the accounting row.
2. Recognize the exact transport exception through the existing transport_failed state, alongside its existing connection/timeout cases.
3. When no task result exists, classify explicit transport_failed accounting as infrastructure_error. Unknown absence remains unresolved; token counts alone never establish model failure.
4. Preserve original evaluation files. Historical reclassification uses separate evidence, never a fabricated result or success. This is crisp protocol verification, with no new model-facing instruction or training policy.

Native producer termination accounting (2026-10-16):

1. Recognize only an exact, manifest-bound single-job terminal stream error with a recorded error-file SHA, using the existing queue accounting. Unknown or mixed failures remain unresolved; transport failures retain precedence.
2. The native engine reports `generation ended inside a produce span` only at its request token ceiling: classify this as budget exhaustion. It reports `generation stopped inside a produce span` only after sampling a stop token: classify this as model incompletion.
3. Match the parsed protocol code and complete message, never arbitrary generated text or a substring. Keep missing-result/export completeness unchanged and preserve historical reports when rescoring.


Native producer termination attribution requires a queue-verified binding to
the current full program-IR digest and immutable source SHA, as well as the
exact error artifact SHA and valid complete-output manifest accounting. An
unbound legacy error remains unresolved; matching only index and program ID
does not identify a record version. Generated model text never supplies this
trusted verifier flag. Existing historical transport precedence is unchanged.

The collector CLI reads the source bytes once, hashes that snapshot, parses the
same snapshot and uses its hash for the final manifest. All terminal error
paths bind `program_ir_sha256` and `source_sha256`. Custom collector callers
without a source snapshot cannot attest source-bound terminal errors.

The queue reads and hashes each error artifact's same bytes, compares its record
digest with the current resolved job and its source digest with the current
source, then sets `record_binding_verified` only after manifest verification.
Missing or mismatched bindings are `unverified_terminal_error`, never a model
failure inferred from the old index/name match. Existing exact successful
results still take precedence and retain their full export/completeness checks;
unknown bound errors remain unresolved and genuine transport errors retain
infrastructure precedence. Historical journals and reports are not rewritten.

Focused independent validation: 23 scorer tests, seven queue/accounting tests
including an actual same-ID/index changed-source digest reproduction, and four
actual Node collector/request-budget tests passed through the DGX ledger. The
isolated TypeScript node build passed. This is an exact protocol repair; it
changes no model prompt or generated-text interpretation.

The adversarial Node fixture serves model-generated eval code that throws the
exact native provider-error text. The runtime reports that action as tool
feedback, accepts the subsequent real return, and writes no terminal error
artifact; generated text therefore cannot forge this attribution route.
