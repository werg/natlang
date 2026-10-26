---
args: {"segment": "ContextSegment", "candidate": "LocaleCandidate", "evidence": "EvidenceItem[]"}
returns: "string[]"
---
Read the candidate independently against the source's actor, action, preconditions, exceptions, negations, quantities and consequences. Use surrounding product context, not back-translation fluency alone. Probe an ordinary and an exception case where changed wording would cause a different action. Return each material difference with its exact segment and evidence; do not accept a changed sense because the target text sounds natural.
