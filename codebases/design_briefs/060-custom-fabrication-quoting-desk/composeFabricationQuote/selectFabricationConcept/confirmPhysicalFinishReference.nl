---
args: { concepts: ShopConcept[], finishes: FinishReference[], samples: Ref[] }
returns: FinishAcceptanceReview
---
Accept a finish only when the concept's exact reference is documented, the customer accepted that reference, and an allowed physical sample or approved reference is present. Keep screen-rendering comparison separate from physical finish approval.
