---
description: The digest of a section and of each of its subsections.
args:
  section: Section
returns: Digest
recursive: structural
ensures:
  - check: "covers(result, section)"
    message: the digest keeps the section's title and has one part per subsection, in order
  - predicate: a digest whose gist states the main point of the section's own text in one sentence
    against: [section]
---
Write the gist of section's own text in one sentence. Digest every subsection with digest, all at once, and return
the title, the gist and the subsection digests as parts, in their order.
