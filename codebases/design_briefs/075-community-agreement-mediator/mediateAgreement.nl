---
description: Mediate a community agreement while preserving each participant's account, consent state and divergent trial outcome.
args: { dossier: MediationDossier; change: MediationChange; mediator: Mediator }
returns: MediationProposal
---
Preserve prior agreement versions and competing participant accounts separately. A disputed recollection is not resolved by majority assertion. Keep each term's acceptance tied to its participant and current evidence.

Bind history = `recoverAgreementHistory(dossier, change)`. Nonresponse remains pending. Where prior terms are disputed, retain every account and identify evidence that could resolve that exact disagreement.

Bind positions = `mapParticipantPositions(dossier, change, history)` to identify each participant's stated interests and term-specific positions without invented motive. Bind options = `findBoundedOptions(dossier, positions, change)` only within the supplied agreement scope. Route legal rights or immediate safety matters to qualified support rather than mediating them.

Bind trial = `designReversibleTrial(dossier, positions, options, change)` only when participants, terms, duration, review trigger, exit process and individual acceptance can be stated. If the trial is absent or any acceptance is pending, keep it proposed and do not review it as an active trial.

When a current accepted trial exists, Bind trialReview = `reviewTrialEvidence(dossier, change, trial)`. Report outcomes per participant; success for some is not universal success. On withdrawal, apply the agreed exit process promptly and stop relying on that participant's consent.

Return `MediationProposal` for `mediator.id` with term-level consent, evidence gaps, qualified referrals, divergent findings and a resumable agreement history. Options do not amend an agreement until required participants accept the same version.
