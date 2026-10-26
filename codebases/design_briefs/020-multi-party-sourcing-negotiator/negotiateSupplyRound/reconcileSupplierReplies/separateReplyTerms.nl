---
args: {"response": "EvidencePage", "packages": "PackageCandidate[]"}
returns: "TermReply[]"
---
Break the supplier response into propositions linked to the exact offered package and term. Distinguish acceptance, refusal, qualification and mere acknowledgment. Preserve explicit dependencies between propositions. Where package identity or signing authority is unclear, retain unconfirmed with the source and the smallest confirmation needed; never expand a narrow yes into acceptance of the entire package.
