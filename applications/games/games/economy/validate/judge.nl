---
description: Judge whether one merchant's intent is well formed. A pass is always well formed; a buy names another merchant, a good and a whole quantity.
args:
  known: string[]
  submission: Submission
returns: Verdict
---
Judge submission.intent, the intent of the merchant submission.actor. known lists the ids of the merchants of the economy.

1. When submission.actor is not in known, the verdict is not ok, reason "the actor is a merchant of this economy".
2. Kind "pass" is ok, reason "a pass names nothing".
3. Kind "buy" is ok when all three hold, and otherwise not ok with the reason of the first that fails:
   - seller is a string in known that differs from submission.actor ("a buy names another merchant of this economy as
     seller");
   - good is a string that starts with a letter and continues with letters, digits, "_" or "-" ("a buy names a good by
     a name of letters, digits, _ and -");
   - quantity is a whole number of at least 1, checked in eval with Number.isSafeInteger ("a buy has a whole quantity
     of at least 1").
   A well-formed buy has reason "a well-formed buy".
4. Any other kind is not ok, reason "the intent is a buy or a pass".

Return actor as submission.actor, ok and reason. Stock and cash are for settlement and play no part here.
