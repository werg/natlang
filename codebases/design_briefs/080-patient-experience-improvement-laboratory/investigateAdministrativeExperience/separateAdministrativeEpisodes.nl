---
args: {"feedback": "EvidenceItem[]", "changes": "EvidenceItem[]", "governance": "EvidenceItem[]"}
returns: "AdministrativeEpisode[]"
---
For each attributed feedback item call segmentMixedComplaint(item, governance). Preserve mixed positive/negative administrative experiences, channel and reported consequences without inferring treatment quality. Separate the minimum clinical concern routing reference from the administrative episode. Deduplicate repeated source versions without treating a repeated complaint as a new person. Missing consent or patient scope remains a linkage question.
