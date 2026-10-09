---
description: Decide whether the customer should hear about what just happened to their order, and write the message.
args:
  report: Report
returns: CheckedOutgoing | null
---
report says what happened: report.before and report.after are the order before and after the event, report.decision is
what was done. Write the customer's message with the stages in your folder; ledger checks it.

1. kind = moment(report). When kind is "none", return null.
2. told = facts(report, kind).
3. draft = compose(kind, told).
4. problem = ledger.checkMessage(report.after, draft). When problem is null, return draft.
5. Otherwise draft = compose(kind, told, problem), and check it with ledger.checkMessage again. When the problem is null,
   return draft; otherwise return null (the customer is told nothing rather than something wrong).
