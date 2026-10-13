---
description: Interpret one submitted response without rebinding the suspended announcement.
args:
  response: IntentReading
  beforeBoard: BoardSnapshot
  source: TurnSource
returns: InterruptResolution
---
1. Establish from the original response evidence that this is an actually submitted legal response.
   Read its effect and target instance against beforeBoard. A merely illustrated ability or proposed
   later action cannot enter this committed response window.
2. Separate controller change, attachment movement, resource payment and zone change. Identity survives
   a control change within the garden; an attachment move changes the artifact's attachedTo field and
   does not itself transfer either creature's control or spend a protection-use token.
3. Identify exactly which source fields this response changes and which it preserves. Construct one
   ObjectChange per affected instance with its actual before object and fully preserved after object.
   Copy witnesses establishing both the object identity and this submitted effect.
4. Respect already-paid costs as source facts. Add resource/token changes only for a source instruction
   making them part of this response now. Preserve missing target identity or disputed legality as
   unresolved; the caller then keeps the pre-response state rather than applying a partial effect.
5. Return responseIntentId, object/resource changes, consumedTokenIds, exact witnesses and unresolved.
   The parent applies them once and retains the original announcement and its captured reference values.
