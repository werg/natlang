# Natlang document publisher

`publish.nl` reads exact passages from a pinned evidence revision, asks natlang
to plan and compose a structured document, checks citation authenticity and
asset/table IDs, then prepares Markdown and HTML from the same tree. Natlang
owns the outline, wording and claim selection. The host owns source bytes,
tables, rendering, and the atomic publication pointer.

`DocumentPublisher` publishes to a local versioned directory and changes one
symlink only after both representations have been written and the evidence
revision has been rechecked. A prepared output is not an external release.
Literal quotes establish provenance only; the semantic relevance of a claim
still requires model evaluation. Visual quality and live teacher quality are
separate gates.
