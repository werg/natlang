---
kind: directory-reducer
args:
  request: TransformationRequest
returns: TransformationExplanation
---
Simplify instructions and API usage for an underpowered coding model using observed failures. Prefer explicit folder.propose, evaluator.check/evaluate, folder.accept and folder.select. Do not memorize answers.
Read the real source and request before editing. Make actual file changes in folder and return changed paths, rationale, and preserved rules. Evaluation and acceptance belong to the caller.

Stay within request.allowedFiles. Preserve request.preserves when supplied and the declared entrypoint. Use request.evidence and any supplied request.checks to test the hypothesis; expected held-out answers are unavailable. Return the actual changed paths, not an intended patch.
