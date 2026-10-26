---
args: { request: SupplierIncidentRequest, policies: PolicyCandidate[] }
returns: PolicySelection
---
Select supplier quality, containment, and customer-notice terms only when accountable authority, effective window, product scope, and order scope cover this incident. Keep uncertain or conflicting precedence explicit; severity alone does not grant a wider hold or notice authority.
