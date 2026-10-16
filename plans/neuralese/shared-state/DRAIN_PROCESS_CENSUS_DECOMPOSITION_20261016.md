# Capture the actual optimizer-owner process tree

The supervisor's existing natural-language policy decides when useful training
should drain. This change is a crisp Linux observation/identity verifier only;
it neither chooses that boundary nor sends signals or changes model arithmetic.

1. Authenticate the launch and observe its actual live PID, boot, start ticks
   and argv. If a launch already binds an identity, require that identity too.
2. Enumerate both all-thread `children` files and `/proc/*/stat` parent-PID
   relationships. Their union determines candidate descendants; an unavailable
   `children` file cannot erase a real parent relationship.
3. Observe each candidate identity and its actual parent, then authenticate the
   parent's continued identity. Reject conflicting parent or reused-PID evidence.
4. Recheck captured live relationships and scan for new unrecorded descendants
   before publication. A moving tree fails closed for a fresh capture; exited
   children remain recorded with their original identity for drain accounting.
5. Persist exact parent/child identity links and observation origins alongside
   the existing immutable launch snapshot. Verify that new receipts cover every
   captured descendant once, rooted at the bound owner. Historical receipts
   preserve their original versioned evidence rather than inventing new facts.

This is a pre-signal observed census, not a claim that userspace can prohibit
future process creation. Existing successful owner termination and all captured
child termination remain mandatory before checkpoint ownership is released.

Validation: ledgered handoff/raw-round/drained-recipe integration passed 37 tests
with one existing CUDA skip (9.73 s, 635 MiB); final handoff suite passed all 24
cases (4.73 s, 554 MiB), including actual nonleader-thread spawning with the
children API absent, PID reuse/reparent/growth rejection and forged archive links.

New drain/2 intents require complete tree evidence. Historical drain/1 receipts
remain readable with their original scope; missing drain/2 links cannot silently
fall back to the historical contract.
