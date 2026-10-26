---
args: {"tasks": "ReaderDecisionTask[]", "units": "OperativeUnit[]", "prior": "ReaderParagraph[]"}
returns: "ReaderParagraph[]"
---
For each task call formulateTaskParagraph(task, itsUnits, prior). Order the explanation around the reader's decisions, introducing necessary background before use. Keep explicit unit IDs with each paragraph and retain technical terms needing explanation. Preserve all material conditions, exceptions and uncertainty. An example must be labeled illustrative and cannot create a new eligibility rule or imply a broader guarantee.
