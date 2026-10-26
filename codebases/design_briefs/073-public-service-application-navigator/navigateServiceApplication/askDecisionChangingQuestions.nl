---
description: Ask only for safe, truthful information that changes a supported service pathway.
args: { dossier: NavigationDossier; change: ServiceChange; pathways: PathwayGraph }
returns: NavigationQuestion[]
---
For each unresolved condition, identify the question whose answer would distinguish supported routes. State acceptable evidence without implying a preferred answer. If a form cannot represent the person's situation, ask for an authorized clarification path instead of forcing an inaccurate field. Do not request information outside the service rule's scope.
