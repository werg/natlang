---
description: Derive competence from charter roots and reject unsupported delegation components.
args:
  archive: Archive
returns: AuthoritySet
---
1. Interpret charter and claimed delegation clauses into CompetenceClaims. Record holder, actKinds,
   territories, interval, onwardDelegation and explicit parent claims; use root=true only for actual charter grants.
2. Keep issuer competence distinct from the substantive act it attempts. A notice-wording grant
   does not include a tariff waiver merely because both appear in the same circular.
3. Initialize AuthorityFrontier with all claims pending, empty proofs/rejections and the archive.
   Invoke deriveDelegation.iterateOn(frontier) with measure pending.length until pending is empty.
4. Each step must remove a decided claim or an unsupported dependency component. Supported proofs
   carry a charter path; unresolved scope produces an explicit rejection of kind unresolved.
5. Return claims, actual derived proofs, rejected components and unresolved reasons. Do not omit
   the reciprocal letters merely because they failed; the result must explain why no tariff power arose.
6. A competence interval clips an order's claimed duration. Preserve the narrower supported interval
   in the authority proof so later norm compilation cannot restore the unsupported extension.
