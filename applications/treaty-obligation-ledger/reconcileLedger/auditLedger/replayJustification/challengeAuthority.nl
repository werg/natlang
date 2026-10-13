---
description: Challenge whether a final normative proof reaches sufficient root competence in the correct scope.
args:
  proof: Justification
  authority: AuthoritySet
  archive: Archive
returns: AuthorityAssessment
---
1. Resolve the proof's authority premise references and trace their parent-proof paths to charter roots.
   Gather the actual act-kind, onward-power, territory and interval claims used by this conclusion.
2. Capture those source clauses and the final normative act. In eval declare a local AuthorityJudgment
   matching AuthorityAssessment, create nl.with<AuthorityJudgment>, and directly await it on this proof path.
3. Ask whether every delegation preserves sufficient scope and whether the final issuer performs
   the permitted act. A notice-wording power cannot support a tariff exemption; a rootless cycle cannot grant power.
4. Consume permitted, narrowedInterval and supportingProofIds as the audit finding. Preserve null
   for genuinely missing source authority; confidence or another valid power does not fill that gap.
5. Return the actual witnessed judgment. This independently challenges the used proof path instead
   of accepting the earlier authority stage's conclusion as its own evidence.
