---
description: Curate a versioned exhibition argument across object evidence, loans, conservation limits, visitor paths and derivatives.
args: { dossier: ExhibitionDossier; change: ExhibitionChange; editor: ExhibitionEditor }
returns: CuratorialProposal
---
Treat provenance, loan status, community restrictions and conservation conditions as binding on every display projection.

Bind constraints = `interpretObjectEvidence(dossier, change)` to bind object claims and display constraints. A withdrawn loan removes that object's active claim support; do not substitute a weaker object without reframing the argument.

Bind arguments = `reframeExhibitionArguments(dossier, change, constraints)` and distinguish claims that remain supported from claims requiring qualification or removal. Preserve provenance limits and competing interpretations.

Bind zones = `designAccessibleVisitorPath(dossier, change, arguments)`. Bind each visitor task to a zone and optional route. Ensure later interpretation carries the context needed when a visitor skips an earlier zone.

Bind labels = `writeQualifiedLabels(dossier, change, arguments, zones)`. Shorten repetition before removing a crucial qualification. Apply community restrictions across galleries, images, online previews and promotional derivatives.

Bind release = `qualifyExhibitionRelease(dossier, change, constraints, arguments, zones, labels)`. Recheck light exposure and duration against current conservation limits. Any changed condition invalidates affected placements and dependent release elements.

Return `CuratorialProposal` for `editor.id` with the complete derivative restriction map, evidence-backed arguments and resumable release state. A draft label is not display or publication permission.
