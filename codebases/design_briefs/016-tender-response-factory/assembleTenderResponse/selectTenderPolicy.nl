---
description: Select the governing tender issue and addendum precedence by issuer authority and scope.
args:{candidates:PolicyCandidate[];tenderId:string}
returns:PolicySelection
---
Match issuing authority, tender identifier, jurisdiction and effective date. Apply only explicit addendum supersession. Preserve unresolved issue sequence or scope precedence; do not select the newest file by timestamp alone.
