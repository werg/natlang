---
description: Decide one intent against the running balances. A pass stays a pass; a buy trades at the seller's offered price, or is rejected with the first condition that fails.
args:
  running: Merchant[]
  submission: Submission
returns: TradeOutcome
---
Decide submission.intent for the merchant submission.actor against running, the balances now.

1. Kind "pass": return { actor, status: "pass" }.
2. Kind "buy": buyer is the merchant of running whose id is submission.actor. seller is the merchant whose id is
   intent.seller, if any. good and quantity are the intent's. Test these conditions in this order and, at the first
   that fails, return { actor, status: "rejected", reason } with the reason shown:
   a. seller exists and its id differs from the actor's: otherwise "seller is not another merchant".
   b. price = seller.offers[good] is a whole number of at least 0: otherwise "seller does not offer the good".
   c. total = quantity * price, computed in eval, is a safe integer (Number.isSafeInteger): otherwise
      "total is not a safe integer".
   d. seller.goods[good] (0 when absent) is at least quantity: otherwise "seller lacks the stock".
   e. buyer.cash is at least total: otherwise "buyer lacks the cash".
3. When every condition holds, return { actor, status: "traded", seller: seller.id, good, quantity, total }.
