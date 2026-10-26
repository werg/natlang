---
description: Select governing cancellation, billing, data and linked-service terms for the named request.
args:{candidates:PolicyCandidate[];requestedOutcome:string}
returns:PolicySelection
---
Apply only current terms whose authority and subscription scope match. Preserve conflicts, date precedence and reseller-specific ownership as questions. Do not let a retention offer or unrelated policy change the requested intent.
