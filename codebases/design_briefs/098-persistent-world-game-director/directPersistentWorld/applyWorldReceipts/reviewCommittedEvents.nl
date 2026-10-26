---
description: Reconcile simulator event content with the exact action, actor, world revision and character visibility.
args: { world: PersistentWorldState; actions: LegalAction[]; receipts: WorldTransitionReceipt[]; readback: WorldSimulatorReadback | null; candidateEvents: WorldEvent[] }
returns: ReceiptEventReview
---
For each candidate event, bind its sourceActionId to exactly one action and its actorId to that action's actorId. The event must appear in the committed receipt's eventIds and in the current readback world at the receipt's newRevision. Keep a privacy-bound event visible only to its listed privateTo recipients; do not use it to update an unlisted character. If the event's content reports an outcome unrelated to its action, or if its action/actor/revision binding is unclear, keep that event unresolved and block the result from being narrated as a world fact. Return the exact accepted events, unresolved event IDs and blockers.
