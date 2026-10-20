# Continue useful training while runtime qualification is deferred

The supervisor's policy is to keep verified finite training moving, retain full
optimizer/RNG/exposure, and perform runtime/task qualification at its declared
readiness boundary. A deferred check grants no readiness and is not a numerical
failure. Actual numerical or runtime failures still require diagnosis.

1. The shared trainer checks model parameters/buffers and optimizer state for
   finite values at its saved reporting boundary. This exact numerical verifier
   is crisp; it changes no weights, optimizer values, RNG or exposure.
2. Store its checkpoint-bound numerical receipt separately from the existing
   runtime qualification receipt. Keep all readiness/completion flags unchanged.
3. The shared supervised dispatcher can publish an earned partial refresh when
   that numerical receipt passes and the authenticated runtime certificate is
   explicitly deferred for the declared on-request schedule. It must reject an
   actual failed runtime check, missing/mismatched evidence, or nonfinite state.
4. Older reports retain their existing meaning. Repairing an old failed bridge
   requires an explicit recovery; never rewrite its failure into success.

All model/stage lines use the same trainer and verifier. No model-facing prompt,
Neuralese format, numerical update law or qualification gate changes here.
