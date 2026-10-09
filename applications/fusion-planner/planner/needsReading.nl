---
args:
  edge: EdgeFacts
returns: ReadNeed
---
Say whether a person or a later check needs to read the value of this edge as text. edge.excerpt, when present, is the part of the orchestrator's instructions that wires the edge; edge.type is the value's declared type; edge.chain names the two functions.

needed is true when any of these holds:
- The value is a diagnostic, an error report, an explanation, a justification or a summary meant for a person.
- The value is a plan, a draft or a proposal that a person or a later check is likely to approve, compare or verify.
- The excerpt says the value is logged, reported, shown or checked.

needed is false when the value is working material only the next function uses: a parse, an analysis, a list of candidates, an intermediate representation, a rewrite of the input.

Answer from the type, the names and the excerpt only; do not guess at facts they do not show. When you cannot tell, answer needed true. reason is one sentence.
