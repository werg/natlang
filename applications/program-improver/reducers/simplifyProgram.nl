---
kind: directory-reducer
args:
  request: TransformationRequest
returns: TransformationExplanation
---
Simplify the native semantic program. Remove unnecessary machinery, make semantic decisions explicit in natlang, retain finite execution and external behavior.
Read the real source and request before editing. Make actual file changes in folder and return changed paths, rationale, and preserved rules. Evaluation and acceptance belong to the caller.

Stay within request.allowedFiles. Preserve request.preserves when supplied and the declared entrypoint. Use request.evidence and any supplied request.checks to test the hypothesis; expected held-out answers are unavailable. Return the actual changed paths, not an intended patch.
