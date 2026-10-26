---
description: Build a reversible historical identity network over approximate dates and time-specific geography.
args: { dossier: LinkageDossier; change: LinkageChange; archivist: LinkageArchivist }
returns: HistoricalLinkageProposal
---
Preserve source names, raw date and place expressions, transcription versions and attributes. Do not replace historic geography with modern identity.

Bind normalized = `normalizeHistoricalObservations(dossier, change)`. Set observations = normalized.observations, dates = normalized.dates, and places = normalized.places to keep raw forms with evidence-backed intervals and period-specific jurisdiction candidates. Approximate dates remain intervals; do not invent point dates.

Bind candidates = `generateIdentityCandidates(dossier, change, observations, dates, places)`. Compare chronology, relationships, locations and attributes. A family name reused across generations is not enough to link identities.

Bind findings = `judgeIdentityEdges(dossier, change, candidates, dates, places)`. Preserve same, different, related and uncertain as distinct findings. Conflicting evidence keeps the edge uncertain.

Bind networkResult = `reconcileLinkageNetwork(dossier, change, findings)`. Set network = networkResult.network and impacts = networkResult.impacts. If a prior merge is unsupported, permit a reversible split and trace affected links and edition statements. A corrected transcription invalidates only links depending on the changed source.

Bind researchEdition = `prepareResearchEdition(dossier, change, network, impacts)` with search scope, unresolved questions and historical geography. Return `HistoricalLinkageProposal` for `archivist.id` with reversible network changes and resumable edge-impact state. A research view is not a declaration of certain identity.
