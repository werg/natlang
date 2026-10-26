---
args: {"use": "DatasetUse", "events": "Evidence[]", "rules": "string[]"}
returns: "ObservableLabel[]"
---
For every proposed label call testLabelObservability(labelDescription, use, events). For observable labels call specifyAnnotationBoundary(labelDescription, events, rules). Keep observable behavior separate from inferred attributes and define explicit uncertainty/adjudication states. Preserve any actual independent-annotator disagreements; do not invent annotator agreement or expected answers to make the rule look reproducible.
