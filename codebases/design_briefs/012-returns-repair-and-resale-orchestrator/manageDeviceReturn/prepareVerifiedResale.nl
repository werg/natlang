---
description: Prepare a truthful resale disclosure only after title, functional test and erasure verification.
args:{scope:GoalContract;policy:PolicySelection;custody:StageResultCustodyEvent;erasure:ErasureVerification[];functionalTests:EvidencePage[];marketplaceRules:PolicyCandidate[]}
returns:StageResultResaleDisclosure
---
Bind releaseChecks = `verifyResalePrerequisites(scope, custody, erasure, functionalTests, marketplaceRules)`. Require a verified title/ownership transfer, complete serial custody, passed device-specific erasure and required functional tests. Unknown or failed erasure quarantines the item. Map known defects and repairs to the current marketplace grade vocabulary; never conceal a defect to achieve a higher grade. Listing remains blocked until every required check is supported.
