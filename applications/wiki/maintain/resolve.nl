---
description: Link resolution. Say what one [[link]] points at on the page now, and whether a section it named has been renamed.
args:
  link: LinkRef
  outline: Outline
  previous: Outline
  block_ids: string[]
returns: LinkStatus
---
Resolve link, a [[link]] in block link.block_id, on the page whose outline is `outline`. block_ids are the IDs of all the
page's blocks (cells included). previous is the outline of the page before the last edit. Work in these steps.

1. Matches. Collect the distinct things link.target names, compared without regard to case: a section of `outline`
   whose id or title equals it (the thing is the section's id), and a block ID in block_ids that equals it (the thing is
   that block ID).
2. One distinct thing: { status: "ok", resolves_to: that thing }. Two or more: { status: "ambiguous", resolves_to: null }.
3. None: look for a section of `previous` whose id or title equals link.target. When there is one, find the section
   of `outline` that starts with the same first block ID. If there is one and its id differs, the section was renamed:
   { status: "renamed", resolves_to: its id }. In every other case: { status: "broken", resolves_to: null }.
4. Return { block_id: link.block_id, target: link.target, status, resolves_to, reason } with reason one sentence for
   the status.
