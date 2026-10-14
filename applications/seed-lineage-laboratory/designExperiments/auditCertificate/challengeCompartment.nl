---
description: Challenge shortcuts that confuse tissue ownership, reciprocal crosses and assay answers.
args:
  source: BankSource
  posterior: Posterior
  predictions: PredictionIndex
returns: AuditFinding[]
---
1. Form one focused challenge for each source distinction that can alter a prediction: maternal coat
   versus adult trait, reciprocal parent role, exact tray versus individual sample, and regime sensitivity.
2. For each challenge bind the actual typed scope/cross/prediction and only its governing clauses.
   Declare a local ChallengeMeaning with passed, reason and clause-keyed quotes, then await a scoped literal.
3. The literal identifies the claimed quantity and causal owner, compares the actual returned
   prediction to those source conditions, and returns true, false or null with the precise uncertainty.
4. Consume every actual result and bind its nonempty exact supporting witnesses to original records.
   An invalid attribution cannot certify a prediction even when its numerical answer appears plausible.
5. Return all AuditFindings, including passed negative controls. Findings diagnose the actual result;
   they do not overwrite a wrong prediction or silently repair the chosen policy.
