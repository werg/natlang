---
args: {"probes": "ConsequenceProbe[]", "sourceAnswers": "ConsequenceAnswer[]", "draftAnswers": "ConsequenceAnswer[]", "paragraphs": "ReaderParagraph[]"}
returns: "FidelityDifference[]"
---
Compare the actual actions and conditions implied by the two independently returned answers for each probe. Equivalent wording may differ, but changed actor, obligation, permission, exception, deadline or uncertainty is material. Identify the affected paragraph and source unit to repair. A concise draft that omits an unless clause fails even if most ordinary cases match. Preserve all disagreements; do not replace the draft reader's answer with the intended source answer.
