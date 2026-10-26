---
args: { items: AssessmentItemDraft[], assessment: AssessmentBuildRequest }
returns: AnswerabilityLedger
---
For each item compare every stated prerequisite and allowed resource with the supplied evidence and task context. Return any absent knowledge or alternate interpretation that changes the answer; classify an item as answerable only when neither makes the result depend on an unstated choice.
