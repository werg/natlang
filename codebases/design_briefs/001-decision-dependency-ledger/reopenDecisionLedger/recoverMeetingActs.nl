---
args: {"sources": "EvidencePage[]", "authority": "PolicyCandidate[]"}
returns: "AttributedAct[]"
---
Read each meeting or correction source chronologically, segmenting when topic or motion changes. For each segment bind acts = classifyMeetingActs(segment, authority). Preserve exact speaker attribution, explicit objections and whether an amendment supersedes a statement. Collect acts without interpreting silence, a suggested motion or a second-hand recap as adoption. Missing source coverage stays a question; duplicated text under another project alias is not a second witness.
