---
args:{leftDuty:DutyFacet;rightDuty:DutyFacet;evidence:EvidencePage[]}
returns:ObligationRelation|null
---
Compare these two duty facets in their exact contract editions. Create a directed depends, excepts, supersedes or conflicts relation only when operative clause language and source context establish it. Record the source-supported scope and both obligation identities. If precedence is ambiguous or the clause editions differ without a valid linkage, return no relation and preserve the question for review.
