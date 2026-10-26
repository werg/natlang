---
args: { request: GrantCycleRequest; reports: GrantReport[]; policies: PolicyCandidate[]; evidence: EvidencePage[]; amendments: EvidencePage[] }
returns: GrantConditionSet
---
Resolve the award and each amendment to the exact grant, funder, project and reporting window. Separate allowed uses, exclusions, spending limits, milestones, submission requirements and effective intervals. Do not let aspirational impact language become a mandatory outcome or a proven claim. Where amendment text does not establish whether a permission applies retroactively, retain the competing readings and block affected period eligibility. Return only attributed conditions plus missing authority and timing questions.