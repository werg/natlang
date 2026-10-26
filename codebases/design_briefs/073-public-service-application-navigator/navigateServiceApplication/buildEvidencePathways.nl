---
description: Match service conditions to explicit current evidence routes, including permitted alternatives.
args: { dossier: NavigationDossier; change: ServiceChange; situation: ServiceSituation[] }
returns: PathwayGraph
---
Call `verifyAlternativeEvidence(dossier, change, situation)` and bind each typed alternative route. For each service condition, read the current rule version and enumerate its standard and alternative evidence routes. Attach each supplied document to only the conditions it can establish. A missing standard document is not a denial when a current alternative route exists. Keep a route pending until the receiving authority verifies it. Revalidate only the pathways touched by an updated rule.
