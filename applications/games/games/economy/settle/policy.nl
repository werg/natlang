---
description: Settle a tick's intents one after the other in the given order, each against the balances the earlier trades left. Returns the outcomes and the entries they amount to.
args:
  merchants: Merchant[]
  ordered: Submission[]
  problem: string
returns: Settlement
uses: [games/economy/ledger]
---
Settle the intents in ordered, in exactly that order, one after the other: each intent sees the cash and goods the
trades before it left. merchants are the balances at the start of the tick. problem is empty, or the check that
rejected an earlier settlement of these intents; settle so that this check holds.

Keep running, a copy of merchants, and two lists: outcomes and entries, both empty. For each submission in ordered:
1. outcome = quote(running, submission). It says pass, rejected with a reason, or traded.
2. made = ledger.entriesOf(outcome). Set running to ledger.post(running, made).
3. Append outcome to outcomes and the items of made to entries.

Return { outcomes, entries }: one outcome per submission, in the order given.
