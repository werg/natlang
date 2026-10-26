---
args: { factory: RegressionFactoryRequest, contracts: BehavioralContract[], tests: RegressionTestSpec[] }
returns: AssertionInvalidationPlan
---
Join changed requirement IDs to assertions whose expected behavior derives from them. Preserve unaffected assertions and prior suite versions. Mark linked assertions invalid until the changed requirement receives a reviewed interpretation and new independent oracle.
