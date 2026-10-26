---
args: {"page": "PublicationPage", "draft": "PatchDraft"}
returns: "CandidatePage"
---
Require page identity, edition and supplied content hash to match the draft's base binding. A mismatch returns text null with an editError and needs a fresh source snapshot. Otherwise bind applied = applyRangeEdits(page.text, draft.edits). Return the exact applied text and error alongside the intended and protected claim keys. This is an in-memory candidate only; no external page has changed.
