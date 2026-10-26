---
args: {"slice": "OwnershipSlice", "reply": "SuccessorAcknowledgment"}
returns: "OwnershipSlice[]"
---
Compare each accepted and declined phrase to the current obligation scope. Require a complete non-overlapping partition; ambiguous boundaries remain unanswered rather than assigned twice. Retain an accepted slice with the named successor and reply evidence, plus a separate residual with acceptedBy null. A scope not mentioned in the reply is unowned, not silently accepted. Preserve the original promise identity and separately confirmed checkpoint.
