---
description: Check resale release prerequisites against exact item identity and marketplace policy.
args:{scope:GoalContract;custody:StageResultCustodyEvent;erasure:ErasureVerification[];functionalTests:EvidencePage[];marketplaceRules:PolicyCandidate[]}
returns:{titleVerified:boolean;custodyComplete:boolean;erasurePassed:boolean;testsPassed:boolean;gradeTerms:string[];blockers:string[];evidenceIds:string[];}
---
Match every check to the requested item serial. Erasure must have passed status, method and verification ID; functional tests must cover all required procedures; custody chain must end with authorized seller; title transfer must be evidenced. Use only applicable marketplace grade terms. Return every unmet condition as a blocker.
