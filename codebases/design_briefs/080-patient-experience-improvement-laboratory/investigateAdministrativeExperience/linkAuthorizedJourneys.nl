---
args: {"episodes": "AdministrativeEpisode[]", "feedback": "EvidenceItem[]", "governance": "EvidenceItem[]"}
returns: "JourneyAssociation[]"
---
For each episode call checkPatientScopedLink(episode, feedback, governance). Use approved identifiers, authorized purpose and actual time windows before linking appointments, referrals or contacts. Group repeated reports only when they describe the same established episode. Keep uncertain same-episode links unresolved instead of counting them as independent experiences or borrowing another patient's events.
