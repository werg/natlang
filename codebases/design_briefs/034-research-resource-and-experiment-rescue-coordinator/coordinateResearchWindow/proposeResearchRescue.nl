---
args: { mandates: ProtocolMandate[], assignments: AssignmentSearch, resources: ResourceAssessment[], snapshot: ResearchOperationsSnapshot }
returns: RescueAlternative[]
---
1. Bind orderChanges = testResequencing(mandates, assignments, resources). It may alter precedence only when the scientific dependency graph permits.
2. Bind deviceChanges = testSubstituteBehavior(mandates, resources, snapshot). Keep each measured-vs-required behavior delta and the authority needed to accept it.
3. Bind windowChanges = testPostponementAuthority(mandates, assignments, snapshot). A new sample window requires a protocol authority that covers that window.
4. Return only alternatives that retain hard predicates or explicitly request the investigator's protocol deviation. Do not report an alternative as approved merely because it is schedulable.
