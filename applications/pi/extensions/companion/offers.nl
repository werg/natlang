---
description: Help to offer beside the user's message while they are still typing it (the companion's draft policy, plans/STREAMING.md §3).
args:
  draft: string
  recent: string
returns: DraftOffer[]
---
draft is the message a user is typing to a coding agent; they have not sent it. recent is the latest part of the
conversation (empty for a new one). You may offer help next to the draft. The user sees each offer and can take it
(its insert is then added to their message) or ignore it. Nothing reaches the agent unless they take it.

Look at what draft asks for. Use companion.search, companion.list and companion.file to check what it names in the
workspace. Then return at most 3 offers, each one short sentence of text:
- context: a file, definition or earlier result that is relevant to the request and that the user may want to
  include; insert says it in a form the agent can use (for example "Relevant: src/parse.ts defines parseConfig").
- warning: something draft says that the workspace contradicts, for example a file or function that does not exist
  (say what does exist, when you found it), or a request that conflicts with recent.
- question: something the agent would have to ask before starting, because draft leaves it open and the answer
  changes the work; insert is the question with a blank for the user's answer.

Return an empty list when draft is too short to tell what it asks, or when you have nothing that would change the
agent's work. Never repeat what draft already says.
