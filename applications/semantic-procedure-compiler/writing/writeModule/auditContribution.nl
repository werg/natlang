---
description: Judge whether emitted policy stages and literal results perform the planned semantic transformation.
args:
  plan: ModulePlan
  text: string
  allPlans: ModulePlan[]
returns: ContributionAudit
---
1. Inventory emitted child calls, inline literal sites, returned fields and state assignments from text.
   Link each to the corresponding planned ownedFields, captures and consumerFields.
2. Capture the proposed source and ownership plans. In eval declare ContributionJudgment with
   substantive:boolean|null, consumedChildFields:string[], consumedLiteralFields:string[], findings:string[]
   and witnesses:{sourceId:string;path:string;quote:string}[]. Construct a scoped NL contribution literal.
3. Directly await it on this module's emitted body. Ask whether each named/inline judgment returns
   information actually used in a later branch/state/output, and whether this parent adds its own
   source-grounded transformation instead of forwarding an unchanged child's final result.
4. Consume the returned substantive flag and field lists in ContributionAudit. A static wording claim
   of consumption is not actual live proof; this audit identifies missing authored data flow only.
5. Name precise repairs in findings: absent typed state handoff, missing unknown branch, ignored
   returned guard or a policy decision moved into crisp code. Do not reward decorative step counts.
6. Record LocalDecision only after the actual literal result. Return path, substantive, consumed
   fields, findings, witnesses and the decision so writing can revise or diagnose the candidate.
