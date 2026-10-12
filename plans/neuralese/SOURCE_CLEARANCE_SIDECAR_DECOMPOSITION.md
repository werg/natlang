# Source clearance sidecar decomposition

The provider child materializer selects one exact, reviewed action per source item. Selection records the source item,
visible-input binding, invocation, and target hash. It does not grant training or trace admission; the corpus registry and
its consumer own that decision.

| Stage | Representation | Reason |
| --- | --- | --- |
| Verify the signed clearance and its program/source pins | Existing TypeScript verifier, extended with exact reference comparison | Hashes, signatures, and byte identity require exact checks. |
| Store clearance evidence | One immutable packet sidecar, pinned by file SHA-256; each row stores a relative path and compact exact-program binding | The large signed evidence is identical across rows. The sidecar preserves it once without losing the per-row binding. |
| Select a child target | Existing reviewed candidate and native materializer path | It already verifies the visible child invocation and exact target. |
| Record final training/trace admission | Corpus registry policy and downstream dataset builder | A successful selection or conversion is not corpus admission. |
| Resolve a sidecar reference | Caller supplies the packet sidecar bytes; existing verifier checks its digest and binding | The validator must fail closed when the sidecar is absent, malformed, or mismatched. |

Packet paths are relative to the packet root. No program, source row, group, invocation, or target binding is moved out of
the selected row's provenance. Generic SFT renderers continue to require `training_admission.approved === true`; they do
not promote candidate selections on their own.
