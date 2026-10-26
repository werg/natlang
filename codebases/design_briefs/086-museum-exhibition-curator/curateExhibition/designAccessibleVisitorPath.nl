---
description: Connect exhibition zones and visitor tasks while ensuring each route carries the context its later claims require.
args: { dossier: ExhibitionDossier; change: ExhibitionChange; arguments: NarrativeSupport[] }
returns: VisitorContextEdge[]
---
Map each zone's claim prerequisites and each supplied accessible or optional route. Call `testSkippedZoneContext(dossier, change, arguments)` for routes that skip a zone. Later labels must repeat only the context necessary to understand their claim; do not require visitors to follow one path to understand essential material.
