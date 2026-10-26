---
args: { issues: FacilityIssueCluster, preferences: PrivacyUpdatePreference[] }
returns: string[]
---
Create one recipient-specific update using only consented fields and the case alias. Suppress names, unit identities, and free-text details outside the recipient's permitted share scope.
