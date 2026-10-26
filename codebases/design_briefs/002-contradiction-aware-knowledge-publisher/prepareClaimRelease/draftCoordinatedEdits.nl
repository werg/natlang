---
args: {"claims": "QualifiedProposition[]", "comparisons": "PropositionComparison[]", "authorities": "PublicationAuthority[]", "desk": "PublicationDesk"}
returns: "PatchDraft[]"
---
Draft only against supplied readable current full text and resolved authority. Use exact UTF-16 ranges with the current substring as expected. Change the smallest conflicting passage, then inspect examples, summaries and cross-links dependent on that meaning. Preserve unrelated claims and local audience framing. Keep archived pages intact; propose an explicit current-guidance redirect. Leave denied-source titles out of replacement text. If a concurrent edition changed, retain only nonconflicting anchors after comparing their context; never silently reuse approval for changed meaning.
