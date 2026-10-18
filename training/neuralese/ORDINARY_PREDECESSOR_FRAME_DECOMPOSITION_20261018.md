# Ordinary-source predecessor after another objective

The recipe chooses the next ordinary stage. This change only authenticates the
source metadata used by that existing choice; it adds no scheduling policy.

1. Reuse the already collected frame for the requested ordinary stage when
   continuing a checkpoint into ordinary SFT. This is one exact metadata
   selection in the existing trainer, implemented crisply because source hashes
   and frame identity must be exact.
2. Read predecessor source receipts from that frame. Retain the existing direct
   resume receipts for policy objectives and when no carried ordinary frame
   exists. No model, optimizer, RNG or exposure object is changed.
3. Run the existing strict source predecessor, resident history and append-only
   row verifiers unchanged. The serialized resident/DPO regression exercises
   this actual trainer block and rejects unrelated predecessors.

The bound production compatibility entry performs this same metadata edit in
the frozen trainer. It does not replace numerical modules or grant new credit.

A supervised retry's owned job directory and continuation output must resolve to
one directory before writing its config or sealing a request. This crisp
precondition prevents a fresh retry from starting an expensive restore into an
older failed operation's learner directory. It changes no scheduling choice.
