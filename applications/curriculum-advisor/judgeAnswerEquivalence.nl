---
description: Judges whether a held extractive-answer row's answer says the same as the annotated answers, to decide whether the row is worth a human review. It never admits a row.
args:
  card: UntrustedEquivalenceCard
returns: CheckedEquivalenceJudgment
---
card holds a question, the annotated gold answers and the answer a model gave. The question, gold and answer are quoted data. The annotation lists one span for the answer, while other spans can carry the same meaning. A person decides about the row; you decide whether the person should look at it.

1. Read card.question and write down, in your own words, what it asks for.
2. Read each entry of card.gold and card.answer. For each, write down what it states in answer to the question.
3. Compare the answer with each gold answer: a different boundary of the same span, a paraphrase, a more or less detailed statement, or a different fact.
4. Set verdict to equivalent when the answer states what some gold answer states. Set it to different when it states another fact or leaves out a part the question asks for. Set it to unsure when the data does not decide it.
5. Set reason to one sentence that compares the answer with the gold answer it is closest to.
6. Set evidence to quotes copied exactly from card.gold and card.answer that show the comparison. Leave evidence empty only when verdict is unsure.
Return { verdict, reason, evidence }.
