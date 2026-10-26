---
args: {"origin": "OperationalSense", "replacement": "OperationalSense", "sources": "EvidencePage[]"}
returns: "SubstitutionWitness[]"
---
For each supported source situation replace only the term's operational interpretation. Bind original = readOperationalConsequences(origin, situation, sources) and substituted = readOperationalConsequences(replacement, situation, sources), where situation is the actual example text. Compare actors, prerequisites, record obligations, escalation and completion, retaining every material difference. Return a witness or concrete counterexample; a claim of similar intent is not a substitution test.
