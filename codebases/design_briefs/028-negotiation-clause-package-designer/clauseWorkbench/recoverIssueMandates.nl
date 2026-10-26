---
args: { mandate: GoalContract; records: ClauseNegotiation[]; evidence: EvidencePage[] }
returns: NegotiationPositionMap
---
For each requested party and issue, bind preferred outcome, authorized fallback, prohibited concession and mandate source. Preserve authority owner and version. A fallback is not consent to trade it, and silence or an earlier draft does not extend mandate. Return supported positions and unresolved authority conflicts.