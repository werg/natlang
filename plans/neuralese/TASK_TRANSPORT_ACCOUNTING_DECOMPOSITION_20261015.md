# Preserve interrupted model-request evidence in task accounting

The existing task verifier must distinguish model failures from infrastructure interruptions. A real saved case finished its first8192-token reply, then its second HTTP request terminated after120.541seconds with TypeError: terminated; no result row was produced.

1. Existing queue accounting reads the exact index/program-bound error file; retain its error, path and SHA in the accounting row.
2. Recognize the exact transport exception through the existing transport_failed state, alongside its existing connection/timeout cases.
3. When no task result exists, classify explicit transport_failed accounting as infrastructure_error. Unknown absence remains unresolved; token counts alone never establish model failure.
4. Preserve original evaluation files. Historical reclassification uses separate evidence, never a fabricated result or success. This is crisp protocol verification, with no new model-facing instruction or training policy.
