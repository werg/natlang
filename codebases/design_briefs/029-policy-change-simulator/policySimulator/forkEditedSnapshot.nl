---
args: { request: PolicyScenarioRequest, edits: EvidencePage[] }
returns: PolicyGuardrail
---
Compare owner edit time and candidate policy hash with the frozen simulation snapshot. If the candidate changed mid-run, preserve the original run under its original hash and require a new fork; never mix runs from different candidate versions.
