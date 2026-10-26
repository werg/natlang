---
description: Resolve names and nicknames into evidence-backed candidate references without guessing identity.
args: { dossier: OralHistoryDossier; change: ArchiveChange; accounts: OralHistoryAccount[] }
returns: OralHistoryReference[]
---
Treat each mentioned name as a reference. Compare supplied biographical, place and relationship evidence within the interview's scope. Keep multiple candidates when a nickname refers to several people. Do not expose a private identity in a public edition while the reference is unresolved or restricted.
