---
args: {"ledger": "DecisionLedger", "inbox": "DecisionInbox"}
returns: "{ publicEntries:string[]; restrictedEvidenceMarkers:string[] }"
---
For the requesting reviewer's established access, project decision meaning, pending question, authorized source citation and uncertainty. If a private source supports a public decision, expose an opaque restricted-evidence marker only; omit its title, excerpt, private actor and revealing inference. Preserve the public decision where its release is independently authorized. Show late corrections with recorded/effective dates and surviving intervals rather than overwriting the history.
