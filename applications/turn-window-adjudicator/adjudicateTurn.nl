---
description: Adjudicate the supplied closed turn through captured references, rewritten bundles and witnessed windows.
kind: directory-reducer
args:
  request: TurnRequest
returns: TurnCertificate
---
1. Call readTurn(folder, request) to obtain actual source roles, original board, interpreted rules and
   submitted actions. Keep this immutable source and its exact coverage throughout the adjudication.
2. Call compileWindows(source). Consume the returned timing, snapshot and finite
   capacity protocol; retain any unresolved consequential ordering before preparing a board transition.
3. Call captureAnnouncements(request, source, protocol) to preserve original target identity and the
   actual scoped controller-reference decisions. This captured object is not regenerated after responses.
4. Call settleTurn(request, source, captured). Its typed frontier resolves the source-authorized work,
   returning actual window snapshots, replacement decisions, simultaneous deltas and triggered effects.
5. Call auditTurn(request, source, captured, settled). Return its TurnCertificate including unresolved
   evidence and defects. The audit uses returned histories; a convenient independently recomputed final
   board cannot substitute for a missing semantic decision or unperformed window.
6. This is a pure fictional turn proposal from a read-only source snapshot. Every effect is returned
   data. Preserve original source object, intent, part and token IDs; generated trace IDs may vary with
   consistent references. The source supplies game meaning and all submitted choices.
