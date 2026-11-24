---
description: Settle a disagreement between a program's expected answer and a strong model's live answer.
model: reviewer
readout: decision
args:
  dispute: Disagreement
returns: '"expected-wrong" | "answer-wrong" | "ambiguous"'
---
dispute.program is a natlang program, dispute.instance one input instance (JSON), dispute.expected the answer its author
checked, and dispute.answer the answer a strong model returned when it ran the program, which the checker rejected with
dispute.issues.

1. Find each place where dispute.answer differs from dispute.expected.
2. For each, run the program's instructions literally on the instance: read each judgment as the function receiving
   only its own arguments would, and do every exact step as its code does.
3. Return "expected-wrong" when the literal run gives dispute.answer (or a third answer) at some difference, so the
   expected answer must change. Return "ambiguous" when a careful reader could defend both values at some
   difference, so the instance or an instruction must change. Return "answer-wrong" when the literal run gives
   dispute.expected at every difference.
