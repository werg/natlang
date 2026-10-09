---
description: Write the subject and body of a customer message from the facts.
args:
  kind: MessageKind
  told: Facts
  problem?: string
returns: Outgoing
---
Write the message for the customer from told. kind is the message's kind and goes into the result.
1. subject: one line of at most 60 characters, without a trailing period, naming the order (told.order) and what
   happened.
2. body: three short sentences or fewer. Greet politely, give told.summary, then told.next. Polite, plain and accurate to
   the facts; written in the order's own words (the order number, the amount) and without any internal operation key.
3. problem, when given, says why an earlier draft was refused: write the draft so that it satisfies it.
Return { kind, subject, body }.
