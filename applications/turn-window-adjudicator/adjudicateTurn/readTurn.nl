---
description: Read the closed turn and separate normative rules, initial state and submitted actions.
kind: directory-reducer
args:
  request: TurnRequest
returns: TurnSource
---
1. Read manifest.json as SourceManifest and require its caseId to equal request.caseId. Read each
   listed original document exactly once, retaining its manifest id, path and complete original text.
2. Inventory bracketed clause identifiers at the beginning of source lines. This is exact source
   framing, not a relevance classifier. Retain the complete original line as the clause's witness.
   If an ID has incompatible text in two documents, preserve that source conflict as unresolved.
3. Call readRules(clauses). Consume its actual rule meanings, controller bases, use tokens and
   context classification. Source-defined effects keep their timing roles even when words overlap.
4. Call readBoard(clauses, rules). Consume actual original object identities and ownership/resource
   state. A rule describing a future return is not evidence that the returned instance already exists.
5. Call readIntents(clauses, request, initialBoard). Consume only actual submitted announcements and
   responses. A hypothetical consequence or a rejected alternative remains source context.
6. Cross-check each submitted target against original instance identities. Preserve the distinction
   between a card template, a current garden instance and an explicitly permitted future instance.
7. Check every original clause ID is covered by actual rules.contexts clauseIds or rules.unitResults
   primaryClauseIds. Failed/unresolved unit results retain coverage but not semantic success. Return the
   original documents/clauses, actual child results and one sourceCoverage row per listed
   document containing every inventoried clause ID. Propagate child unresolved findings with their source
   identities. This stage cannot settle controller changes or apply replacement effects during inventory.
