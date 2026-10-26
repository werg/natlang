---
args: { prior: NegotiationCheckpoint | null, request: NegotiationRequest, dossier: NegotiationDossier, proposal: NegotiationProposal }
returns: NegotiationCheckpoint
---
For a first negotiation create revision 1 bound to the request case and exact baseline version. Record the current proposal edition, pending confirmation identities, accepted scenario/schema versions and affected interface reviews. For a repeated dossier version and same agreement evidence, return the checkpoint unchanged. On a changed dossier, append the proposal; preserve prior signed positions and clauses, and reopen only requirements whose position, authority, or acceptance scenario changed. Count a requirement as confirmed only when the dossier contains the source-bound authorized response for that exact requirement version. Retain unresolved confirmations and successor authority questions. Advance once for changed evidence or proposal, never for duplicate replay.
