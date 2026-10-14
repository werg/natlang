---
description: Interpret owned speech units without promoting embedded quotes to new promises.
args:
  turn: TurnRecord
  tree: QuoteTree
  references: ReferenceBindings
  source: DialogueSource
returns: ActReading[]
---
1. Select each node's own speech content after excluding child interiors. Retain immediate-child meaning
   as context for a reporting act, without assigning the child's original promise to the outer reporter.
2. For each semantic speech unit call judgeIllocution(node, turn, references, source). Consume the actual
   returned performed status, kind, propositions, polarity, issuer, beneficiaries, activation conditions, consequent predicates and endorsement.
3. Bind actId as a generated identifier and spanId to this actual node. Speaker/time come from the
   node's actual attributed frame. Copy originalTurnId from the actual node; parent cannot replace a quoted Jon issuer with Neri or Mira.
4. Validate returned proposition IDs against the visible source vocabulary and actor IDs against actual
   participants. A genuinely unknown proposition remains unresolved prose, not a fabricated canonical ID.
5. Require nonempty exact supporting quote for each nonempty act contribution. Bind original witnesses;
   conditions keep their separate source support and exact version/object references.
6. A depicted quoted original keeps performed false: its force may be promise without creating a
   new commitment in the forwarding episode. The original performed act is linked through originalTurnId.
   Retain source-established ironic complaint as irony, without adding its literal praise as an endorsed
   belief. Without a decisive tone frame retain the literal's actual alternatives and unresolved act.
7. Return actual act readings, preserving separate promise/report/correction/acknowledgment units when
   one turn contains several. Exact duplicate acts may merge only with identical frame and source scope.
