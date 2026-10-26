---
args: { hypotheses: HypothesisSet[], input: LearnerSessionInput }
returns: LearningIntervention[]
---
Select an explanation that has been independently checked by a subject-matter reviewer and addresses only the currently supported hypothesis. If no checked explanation exists, ask for review rather than generating instructional content as fact.
