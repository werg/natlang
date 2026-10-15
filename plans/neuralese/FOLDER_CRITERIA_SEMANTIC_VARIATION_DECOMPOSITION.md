# Folder criterion semantic variation

The current directory reducer asks `archive_tickets` to read a criterion and
ticket files, define one inline natural-language Boolean judge, apply it to
each ticket, move qualifying files and return their names. This is already a
natlang program. The first live variant changed the root's instruction but
kept four short ticket templates. Many exact teachers then replaced the child
judgment with a JavaScript keyword check. Exact file effects alone do not make
that process suitable for SFT.

## End-to-end units

1. `archive_tickets` is the named directory reducer. It reads `criteria.md`
   and the ticket files and performs file moves using the child decisions.
2. Its ad hoc `nl<boolean>` child is the semantic boundary. It decides whether
   the *current* issue was resolved and the customer confirmed that fix, or
   whether the agreed closure leaves no work under the criterion. It returns a
   Boolean for one ticket. Its instruction must bind the criterion and ticket
   text, including which update supersedes which, rather than encode a keyword
   program.
3. The folder/files service is crisp because it implements exact outside-world
   reads and moves. The fixture builder is crisp because it constructs a pinned
   world and exact oracle from authored category metadata. The oracle never
   infers truth from text by keyword; it checks the known authored outcome.
4. The independent action-shape auditor is crisp because it verifies an exact
   training contract: the child must not substitute literal string/regex tests
   for its semantic judgment. It does not claim semantic correctness of other
   child answers.

## Evaluation before adoption

Prepare a version-3 fixture variant in the existing folder source generator,
but keep it out of the normal family registry. Its ticket updates reuse words
such as "resolved", "customer", "confirmed" and "works" in both qualifying
and nonqualifying cases. The latest update and its referent decide the answer.
Run about 48 live Luna cases, inspect exact outcomes, child return paths,
literal-check bypasses, request count and time, then decide whether to replace
the canonical version 2 source. The trial has no training admission. The
currently running 48-case wording trial is separate and changes only the root
instruction on the old ticket fixtures.

## Trial decision, 2026-10-15

The 48-case semantic-world trial closed with 46 exact outcomes. Only 11 of
those exact trajectories passed the child process audit; 35 used a literal
substring or regular-expression check in a child `eval` to stand in for the
semantic judgment. The source is not adopted for normal generation or training.
The immutable source hash is `261ca90122dccf3ae3601037b3ce97f0f70558e9c997ddd3425d0c6324100471`;
the per-case evidence and decision are in
`/srv/storage/natlang-artifacts/free-provider-generation-20261013-v1/folder-criteria-semantic-world-20261015-v1/quality-comparison.json`
(SHA-256 `b6c9acdc393e24a2849ccbdddb730d83c9d2f4396460e9bc80270572b8df43f7`).

The direct-boundary wording trial on the old fixture had 46 exact and 14
process-passing roots; its baseline had 44 exact and 10 process-passing roots.
The semantic-world variant did not improve the useful yield. The short,
repeated ticket templates still made literal matching attractive. Keep the
canonical v2 source at its present two-case cap and hold every bypassed exact
root. The next source trial should use diverse real-text folder items already
supported by `folder-families.mjs`, or a separately qualified direct finite
natural-language readout that makes the child return a typed decision after
seeing its inputs. Any model-facing change needs a fresh live sample before
adoption.
