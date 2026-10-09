---
description: Proposes, for admission reasons that no rule classifies, the category and next action of the closest existing rule category and a prefix that would match them. Advisory only; a person adds accepted rules.
args:
  bucket: UntrustedBucketEntry[]
  categories: RuleCategory[]
returns: CheckedRuleProposals
---
bucket lists admission reasons that no rule classifies yet. Each has a reason (quoted data), a count and example rows. categories are the existing rule categories, each with its action and example reasons. Propose a rule for each reason in bucket.

For each entry of bucket, in order:
1. Read the reason. A reason often has a prefix before a colon; the prefix names its class.
2. Compare the reason with the example_reasons of every category and pick the category whose reasons it resembles in meaning.
3. Set next_action to the action of that category. When no category resembles the reason, set category to unclassified_review_pending, set next_action to retain_raw_evidence_and_review_before_training, and say so in rationale.
4. Set match_prefix to the longest prefix of the reason that is shared with the other reasons of bucket that belong to the same category, and not shared with any example reason of a different category.
5. Set rationale to one sentence that compares the reason with the example reasons of the category you chose.
Return { proposals } with one proposal per entry, each { reason, category, next_action, match_prefix, rationale }, where reason is the entry's reason copied exactly.
