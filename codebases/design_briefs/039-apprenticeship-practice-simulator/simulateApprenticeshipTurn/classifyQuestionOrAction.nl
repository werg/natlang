---
args: { utterance: string }
returns: UtteranceKind
---
Identify a request for information, a proposed world operation, or a wording that could mean either. Preserve the actual target and intent as separate fields. If the difference changes a consequence, return ambiguous and leave the world untouched.
