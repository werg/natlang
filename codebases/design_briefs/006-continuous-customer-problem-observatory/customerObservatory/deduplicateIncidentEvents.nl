---
args: { episodes: EpisodeRegister, identities: IdentityVersionLedger, outages: EvidencePage[] }
returns: IncidentDedupLedger
---
Group copied messages and follow-up tickets only when customer, event and time window identify the same incident. Keep a repeated failure as a new incident when it occurred on a distinct occasion. During one outage, distinguish incoming ticket volume from independent discovery and from unique affected customers; do not merge incompatible incidents transitively.
