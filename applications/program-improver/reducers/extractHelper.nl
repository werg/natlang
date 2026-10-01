---
kind: directory-reducer
args:
  request: TransformationRequest
returns: TransformationExplanation
---
Extract a repeated semantic judgment into a typed named natural-language helper with a companion folder. Retain external behavior and finite control flow.
Read the real source and request before editing. Make actual file changes in folder and return changed paths, rationale, and preserved rules. Evaluation and acceptance belong to the caller.

Stay within request.allowedFiles. Preserve request.preserves when supplied and the declared entrypoint. Use request.evidence and any supplied request.checks to test the hypothesis; expected held-out answers are unavailable. Return the actual changed paths, not an intended patch.
