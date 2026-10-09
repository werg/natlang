---
name: judge-against-criteria
description: Use for eligibility, classification or filtering under explicit criteria and typed record scope.
summary: Apply the stated rule to evidence for the identified entity; treat typed IDs as scope and require an ID match only when the criterion says so.
---

Read the output contract, each item's exact criterion, and its supplied evidence before deciding. Identify the facts that would satisfy the criterion and the evidence that would defeat it.

When typed arguments or captures identify the entity whose evidence is supplied, treat that identity as the evidence scope. A fact in the supplied evidence belongs to that entity unless the record says otherwise. Do not require the entity's identifier to be repeated in evidence text unless the criterion explicitly requires an identifier match. When it does, compare the requested and supplied identifiers exactly.

Check every stated condition against relevant evidence. Preserve negation, dates, thresholds, current versus historical status, and contradictions. Combine the checks exactly as the rule directs. Do not add predicates, prerequisites, keyword tests, or regular-expression gates, and do not import facts from another entity or record.

When a rule requires all or both of multiple conditions, treat it as a conjunction: every required condition must be supported for the item to qualify. If the evidence affirmatively states that at least one required condition failed (for example, “P failed or Q failed”), the full conjunction is false even when the evidence does not identify which failure occurred. Keep the cause unresolved if needed, but do not make the overall eligibility result unresolved solely because the failing conjunct is not localized. Distinguish uncertainty about which defect applies from uncertainty about whether at least one required condition failed. Apply this only when the evidence asserts the disjunction; a possibility or unresolved alternative is not an established failure.

Keep contradicted conditions distinct from missing facts. Use an unresolved or missing-evidence result when the output contract allows it; otherwise follow its stated rule for missing information. Use false when the criterion is not met. A negative judgment is a successful result, not a blocker.

Do not classify solely by a keyword or invent unseen fields. Compute the final filtered set and its keys from completed judgments in eval. Delegate useful semantic work when needed, and reuse its results.
