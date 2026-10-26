---
args:{prior:CloseCheckpoint|null;request:CloseCycleRequest;books:CloseSourceBook;scope:CloseAssertionScope;breaks:CloseBreakLedger|null;tasks:CloseTaskQueue|null;conclusion:CloseConclusionDraft|null;signoff:CloseSignoffReview|null}
returns:CloseCheckpoint
---
Preserve each assertion’s entity, period, account range and source edition. Append late journal receipts and reopen only workpapers whose source lines changed. A regenerated report with equal totals does not restore changed source-line provenance. Keep formula failures quarantined until the exact formula/input/output receipt passes. Preserve open break ownership and deadline history; a successor inherits the task without changing the original preparer. A signoff applies only to the exact conclusion and source-line versions. Duplicate source and receipt replay does not advance revision.
