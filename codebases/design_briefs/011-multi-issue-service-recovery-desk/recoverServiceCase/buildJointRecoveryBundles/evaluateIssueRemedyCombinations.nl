---
description: Check candidate remedy bundles against entitlements, amount authority and shared-loss identity.
args:{issues:ServiceIssue[];entitlements:EntitlementFinding[];ledger:EvidencePage[];choices:EvidencePage[]}
returns:{bundles:RemedyBundle[];duplicateCredits:string[];outsideAuthority:string[];expiredReservations:string[];}
---
Group credits only when supplied evidence shows the same loss. Retain separate rights for distinct issues. Check exact amount and exception caps from the authoritative ledger. Reject expired inventory reservations and preserve alternatives that need customer choice. Keep unsupported bundles unresolved with the limiting evidence identifier.
