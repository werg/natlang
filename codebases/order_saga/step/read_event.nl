---
args:
  text: Text
returns: Kind
---
Read the order event. A customer placing a new order means start; confirmation of
successful payment means paid; an unsuccessful shipment means failed; confirmation
of completed shipping means sent; a request to cancel means cancel. If none of
these is established by the text, report a blocker rather than inventing an event.
