---
description: Prepare a public edition from only consented source spans and test combinations for indirect disclosure.
args: { dossier: OralHistoryDossier; change: ArchiveChange; accounts: OralHistoryAccount[]; references: OralHistoryReference[]; comparisons: RecollectionComparison[]; consent: ConsentSpan[] }
returns: OralHistoryEdition
---
Call `testCombinedDetailDisclosure(dossier, change, accounts, references, consent)` on the proposed quotations and summaries. A non-identifying name does not make a rare date/place/relationship combination safe. Remove or generalize restricted details only within the consent and archive policy; preserve the source account in restricted storage. Bind each included derivative to all authorizing consent spans and the source transcript version.
