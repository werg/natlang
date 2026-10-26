---
args: { request: PolicyScenarioRequest, versions: PolicyCandidate[] }
returns: PolicyDeltaSet
---
Resolve the named baseline and candidate policy versions by authority, scope, and effective interval. Bind each changed clause to its condition and consequence, enumerate safeguards that remain textually and operationally intact, and retain unresolved precedence. A prose summary that cannot be anchored to both version identities is not a usable delta.
