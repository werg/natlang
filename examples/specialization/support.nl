---
description: Answer a support request about an order.
args:
  request: string
returns: string
---
Answer a customer's support request about an order.

- When the request asks to refund an order and gives its number, refund it with orders.refund(number) and answer "refunded <number>".
- When the request asks where an order is or what its status is and gives its number, look it up with orders.lookup(number) and answer with the status it reports.
- Otherwise answer in one short sentence, without calling a service.
