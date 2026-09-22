---
description: The date an email asks something for, as written; "" if it names none.
args:
  email: string
returns: string
---
If the email in `args/email` names a date or day on which something happens or is due, answer with that date exactly
as the email writes it ("Friday 14 March"). If it names none, answer with the empty text.
