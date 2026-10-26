---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftConsumerDependency
---
Construct a ConsumerDependency draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Relate integrations, customer configurations, examples, and runbooks to the changed behavior. Identify implicit reliance such as ordering or retry assumptions, and preserve uncertain dependencies for targeted replay rather than declaring them unaffected.
