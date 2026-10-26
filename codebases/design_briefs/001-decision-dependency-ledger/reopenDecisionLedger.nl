---
args: {"inbox": "DecisionInbox", "ledger": "DecisionLedger | null"}
returns: "LedgerReview"
---
1. Let prior be the supplied ledger or an empty ledger at revision zero. Bind acts = recoverMeetingActs(inbox.meetingAndCorrectionSources, inbox.authority). Read actual source editions; a copied project note is one origin, even after a project rename.
2. Bind decisions = reconstructAdoptions(acts, inbox.cases, prior.decisions). Keep historically adopted decisions and unclosed objections separate; do not invent acceptance or dates.
3. Bind premises = discoverDecisionPremises(decisions, inbox.meetingAndCorrectionSources, prior.premises). Only supported and owner-confirmed necessary links can force reconsideration. Inferred links produce questions.
4. Bind changes = comparePremiseCorrections(premises, prior, acts, inbox). Preserve recorded and effective time. If there is no supported contradiction or narrowing, keep frontier null and retain existing pending questions; editorial changes do not generate owner work.
5. For a supported affected premise bind frontier = walkReconsiderationFrontier(decisions, premises, changes, prior.questions). Its necessary, supporting, independent and cyclic portions remain distinct. Bind routed = routeCurrentReviewOwners(frontier.questions, decisions, inbox); replace only those questions' routing, never the decision's historical author.
6. Construct the next ledger by appending new source identities and changes, keeping prior resolutions and unaffected questions. Bind view = projectRestrictedLedger(nextLedger, inbox). Return its publicEntries/restrictedEvidenceMarkers, the actual frontier or null, source and authority questions, and notification intentions only for new or changed routed questions. No notification becomes delivered without a supplied matching receipt.
