---
description: Reconcile oral-history accounts as a consent-scoped event graph while preserving speaker-specific recollections.
args: { dossier: OralHistoryDossier; change: ArchiveChange; archivist: Archivist }
returns: OralArchiveProposal
---
Preserve transcript and speaker accounts with exact source spans and certainty language. Do not merge conflicting recollections into one asserted chronology.

Bind accounts = `segmentSpeakerAccounts(dossier, change)` and bind every account to its interview span. Bind consent = `reconcileConsentState(dossier, change, accounts)` before using any account for follow-up or publication.

If a speaker withdraws one anecdote, Bind withdrawal = `applyScopedWithdrawal(dossier, change, accounts)` immediately. Retract only quotations and summaries that depend on that span; retain unrelated material whose consent remains current.

Bind references = `resolveMentionedReferences(dossier, change, accounts)`. A nickname may retain multiple candidate identities; do not guess.

Bind comparisons = `reconcileRecalledEvents(dossier, change, accounts, references)`. Keep conflicting dates and contexts side by side. Do not treat one account as authoritative unless the supplied record establishes that role.

Bind questions = `formNonLeadingFollowups(dossier, change, comparisons)` only for questions that could clarify context without suggesting the desired answer. Do not contact a speaker when consent excludes follow-up.

Bind edition = `prepareConsentBoundEdition(dossier, change, accounts, references, comparisons, consent)`. Review combined rare details for indirect disclosure. Bind each quotation and summary to its source span and consent version.

Return `OralArchiveProposal` for `archivist.id` with unresolved identities, competing accounts, scoped retractions, disclosure findings and resumable derivative lineage. A draft edition is not a publication.
