---
description: Does this ticket need attention within the hour?
args:
  ticket: Text
returns: Bool
---
A ticket is urgent when many customers are affected right now, money is being lost, or security is at risk.
Requests, suggestions and cosmetic problems are not urgent. Answer true or false for `args/ticket`.
