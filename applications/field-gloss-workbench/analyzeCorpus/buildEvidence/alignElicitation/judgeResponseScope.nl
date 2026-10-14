---
description: Judge asserted versus merely depicted fields for a scoped elicitation.
args:
  utterance: UtteranceReading
  session: SessionReading | null
  situation: SituationReading | null
  notes: NoteReading[]
  rules: SourceClause[]
returns: ResponseScope
---
1. Directly invoke a typed scoped NL ResponseScope literal over the actual utterance, prompt/session, situation, effective notes and governing source rules. No grammar candidate or exact expected result enters scope.
2. Identify the prompt's requested distinction and the consultant's actual answer/judgment. List each semantic field actually licensed for comparison in assertedFields.
3. List depicted but unasked fields in excludedFields, including an unexamined future horizon or a scene participant irrelevant to the stated contrast. A complete scene is not automatically a complete utterance assertion.
4. Distinguish accepted language form from accepted proposed gloss. Return only actual original judgmentNoteIds that support this comparison; a student's proposal alone supplies neither verdict.
5. If the response changes the prompt or contrast set explicitly, retain that actual shift in reason/support. If the shift is uncertain, scope.status stays unresolved rather than artificially agreeing with the scene.
6. An established unusable protocol returns unusable; missing evidence returns unresolved. Unknown is not a negative acceptability answer.
7. Return actual field scope, original witness quotes and uncertainty. The parent consumes these fields as the test boundary; it cannot add extra discriminatory scene facts later.
