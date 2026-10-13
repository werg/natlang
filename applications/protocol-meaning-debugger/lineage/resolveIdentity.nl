---
description: Judge definition identity under revision, role and execution context.
args:
  left: DefinitionClause
  right: DefinitionClause
  index: EvidenceIndex
returns: AliasDecision
---
1. Capture both owner/role declarations and explicit revision/hash/binding evidence. Equal filenames
   or common words are candidate clues; physical artifact identity and semantic equivalence are distinct.
2. In eval declare local IdentityJudgment with relation: 'same' | 'different' | 'unresolved', reason:
   string, witnesses: {sourceId:string;path:string;quote:string}[]. Construct nl.with<IdentityJudgment>
   over changing input {leftId:left.id,rightId:right.id,leftMeaning:left.meaning,rightMeaning:right.meaning}.
3. Its instructions are: decide whether these references identify the same selected definition under
   captured ownership and revision rules; separate an explicit alias from a historical same-name record;
   retain missing binding evidence as unresolved and cite the exact source establishing identity.
4. Directly await the literal. Consume its relation as the AliasDecision and preserve its witnesses.
   Do not replace an unresolved identity with a prose-similarity threshold.
5. Include localDecision with the actual site and pair input IDs and the consumed reason. The caller
   uses this result to choose definition links; record it only after the literal actually returns.
6. If the pair is different but semantically equivalent, retain different identity. Proposition comparison
   belongs to compare; collapsing identity would lose provenance of a later repair.
7. Return leftId, rightId, relation, witnesses and localDecision. The decision can distinguish the same
   term used in different roles without declaring either definition numerically wrong.
