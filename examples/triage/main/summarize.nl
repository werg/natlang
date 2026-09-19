---
description: One short paragraph about a set of urgent tickets.
args:
  tickets: Text[]
returns: Text
uses:
  word_count: ../../std/word_count
---
function summarize(tickets) -> Text

  draft = write one paragraph saying what is going wrong across `tickets`, most severe first
  repeat at most 3 times, until word_count(draft) <= 60:
      draft = shorten(draft)
  return draft
