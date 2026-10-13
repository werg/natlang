---
description: Independently challenge one window snapshot choice and semantic consequence timing.
args:
  window: WindowTrace
  windows: WindowTrace[]
  captured: CapturedTurn
  source: TurnSource
returns: TurnFinding[]
---
1. Gather this window's actual predecessor, causal parent, immutable before snapshot, replacement
   history, batch and trigger decisions. Include source timing and capacity clauses with exact witnesses.
2. Capture this focused context. In eval declare a local TimingJudgment with status:'defended'|'defect'|
   'unresolved', reason:string, implicatedSlotIds:string[] and evidence:{clauseId:string;exactQuote:string}[].
   Construct a scoped NL literal that challenges the claimed timing and snapshot semantics.
3. Directly await it on the actual window. Ask whether responses precede preparation, replacement order
   uses current rewritten quantities, sibling conditions share one before-board, departures follow
   simultaneous resolved damage, and queued consequences preserve their source controller references.
4. Consume its actual judgment. Bind original source IDs/paths from parent-held clauses and check
   nonempty exact quotes; verify implicatedSlotIds refer to this real trace or its actual dependencies.
5. Return the corresponding TurnFinding. A literal failure is unresolved audit work and cannot be
   replaced by a generic defended statement; a diagnosed error remains visible in the certificate.
