---
args: { utterance: string }
returns: UtteranceKind
---
Classify the communicative act from the learner's words and surrounding turn context. Preserve both the literal request and the intended target. When question and command readings have different consequences, classify as ambiguous and require clarification rather than choosing the more executable reading.
