import { is_short } from "./summarize/is_short";
import { shorten } from "./summarize/shorten";
---
description: One short paragraph about a set of urgent tickets.
args:
  tickets: Text[]
returns: Text
---
function summarize(tickets) -> Text

  draft = write one paragraph saying what is going wrong across `tickets`, most severe first
  repeat at most 3 times, until is_short(draft):
      draft = shorten(draft)
  return draft
