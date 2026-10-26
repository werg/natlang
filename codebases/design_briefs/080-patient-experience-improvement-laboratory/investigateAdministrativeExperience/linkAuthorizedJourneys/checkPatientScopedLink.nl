---
args: {"episode": "AdministrativeEpisode", "feedback": "EvidenceItem[]", "governance": "EvidenceItem[]"}
returns: "JourneyAssociation"
---
Verify patient-scope identity and permission before using any candidate journey event. A similar name, obstacle or appointment date is not sufficient identity evidence. Restrict the association to the permitted service/period and preserve the exact event IDs. If scope or identity cannot be established, return no cross-record link and a targeted authorized confirmation request.
