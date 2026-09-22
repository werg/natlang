import { is_short } from "./summarize/is_short";
import { shorten } from "./summarize/shorten";
---
description: One short paragraph about a set of urgent tickets.
args:
  tickets: string[]
returns: string
---
function summarize(tickets) -> string

  draft = write one paragraph saying what is going wrong across `tickets`, most severe first
  repeat at most 3 times, until is_short(draft):
      draft = shorten(draft)
  return draft
