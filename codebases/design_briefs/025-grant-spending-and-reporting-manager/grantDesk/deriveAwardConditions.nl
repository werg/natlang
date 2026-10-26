---
args: { request: GrantCycleRequest; reports: GrantReport[]; policies: PolicyCandidate[]; evidence: EvidencePage[]; amendments: EvidencePage[]; amendmentWindow: GrantGuardrail }
returns: GrantConditionSet
---
Resolve the award and each amendment to the exact grant, funder, project and reporting window. Separate allowed uses, exclusions, spending limits, milestones, submission requirements and effective intervals. Do not let aspirational impact language become a mandatory outcome or a proven claim. Carry every affected expense and blocker from amendmentWindow into unresolvedEffectiveDates and unresolvedConditions; do not allow those expenses into conditions eligible for earlier periods. Return only attributed conditions plus missing authority and timing questions.
