---
args: { interventions: RestorationIntervention[], permissions: FieldAccessPermission[] }
returns: FieldActionAuthorization
---
Authorize a field task only when the permission is current for its area and explicitly permits its action. Expired access stops affected tasks immediately while unrelated permitted areas remain scheduled; list unresolved owner evidence.
