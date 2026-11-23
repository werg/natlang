---
description: The bottom-up digest of a section integrated with its subsections.
args:
  section: Section
returns: Digest
recursive: structural
ensures:
  - check: "covers(result, section)"
    message: the digest keeps the section's title and has one part per subsection, in order
  - predicate: a digest whose gist states the main point of the section's own text and integrates any key findings from its subsections in one sentence
    against: [section]
---
First digest every subsection in section.sections with digest, concurrently, to obtain parts in their original order.
Then write a one-sentence gist for section that states the main point of section.text and synthesizes any critical
findings from parts (so a parent digest summarizes the whole subtree, not only its introductory text). Return
{ title: section.title, gist, parts }.
