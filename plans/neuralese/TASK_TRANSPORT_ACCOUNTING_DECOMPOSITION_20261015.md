# Preserve interrupted model-request evidence in task accounting

The existing task verifier must distinguish model failures from infrastructure interruptions. A real saved case finished its first8192-token reply, then its second HTTP request terminated after120.541seconds with TypeError: terminated; no result row was produced.

1. Existing queue accounting reads the exact index/program-bound error file; retain its error, path and SHA in the accounting row.
2. Recognize the exact transport exception through the existing transport_failed state, alongside its existing connection/timeout cases.
3. When no task result exists, classify explicit transport_failed accounting as infrastructure_error. Unknown absence remains unresolved; token counts alone never establish model failure.
4. Preserve original evaluation files. Historical reclassification uses separate evidence, never a fabricated result or success. This is crisp protocol verification, with no new model-facing instruction or training policy.

Native producer termination accounting (2026-10-16):

1. Recognize only an exact, manifest-bound single-job terminal stream error with a recorded error-file SHA, using the existing queue accounting. Unknown or mixed failures remain unresolved; transport failures retain precedence.
2. The native engine reports `generation ended inside a produce span` only at its request token ceiling: classify this as budget exhaustion. It reports `generation stopped inside a produce span` only after sampling a stop token: classify this as model incompletion.
3. Match the parsed protocol code and complete message, never arbitrary generated text or a substring. Keep missing-result/export completeness unchanged and preserve historical reports when rescoring.
