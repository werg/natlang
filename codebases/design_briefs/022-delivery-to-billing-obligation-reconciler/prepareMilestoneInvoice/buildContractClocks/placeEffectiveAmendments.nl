---
args: {"conditions": "ObligationClock[]", "prior": "ObligationClock[]", "sources": "EvidencePage[]"}
returns: "ObligationClock[]"
---
Locate each authorized amendment's scope and effective interval. Split a schedule interval only where the amendment actually changes that obligation. Preserve prior earned rights under their original schedule; apply future rates only to future qualifying events. Keep both the recorded date and effective date of a late amendment. Return an unresolved overlap when neither precedence nor event time is established.
