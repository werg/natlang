---
args: {"inbox": "ExperienceInbox", "studyState": "ExperienceStudyState | null"}
returns: "ExperienceInvestigation"
---
1. Bind prior to studyState or an empty study revision. If changed feedback includes an authorized consent withdrawal, call removeAttributableDerivatives(inbox.changes, prior, inbox.governance) before producing any new public findings. Preserve only permitted restricted audit references and actual removal receipts.
2. Bind episodes = separateAdministrativeEpisodes(inbox.feedback, inbox.changes, inbox.governance). Route clinical concerns through the supplied clinical process using only the necessary restricted information. A pending routing intention is not a triage receipt; do not interpret symptoms, diagnose, or claim the clinical concern was handled.
3. Bind associations = linkAuthorizedJourneys(episodes, inbox.feedback, inbox.governance). Link only within established authorized patient scope and time windows. Repeated reports of one episode remain one experience; an ambiguous link cannot expose another person's record.
4. Bind mechanisms = investigateRivalMechanisms(episodes, associations, prior.mechanisms). Keep administrative mechanisms provisional with contrary cases and a discriminating next observation. If no actual trial outcome artifacts exist, bind trials = designAdministrativeTrials(mechanisms, inbox.governance, prior.trials) and return their evidence-collection work without inventing results.
5. For each trial with actual supplied metrics call adjudicateAdministrativeOutcome(trial, itsMetrics, episodes, associations), collecting the returned outcomes. Preserve severe minority barriers and downstream team burden even if aggregate access appears better.
6. Return an incremented study state preserving actual implementation/routing receipts, counterevidence and unresolved consent actions. State only supported administrative findings; neither fewer complaints nor a service workflow change establishes a clinical benefit.
