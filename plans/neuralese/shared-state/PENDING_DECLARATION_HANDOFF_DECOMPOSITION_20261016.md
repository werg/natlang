# Pending declaration handoffs

The existing recipe says that admitted long rows remain mandatory while useful
active rows train. Adding these obligations at a drained runtime handoff must
not require an artificial active-source append or rewrite its predecessor.

1. The owner/admission declaration remains policy: its original row identities,
   bindings and visit quotas define the pending work.
2. The recipe's existing exact verifier admits only a changed pending declaration
   with every other resolved input unchanged. It hashes all declared artifacts
   and checks that earlier obligations and dependencies remain exact subsets.
3. Existing recipe handoff, checkpoint restoration and continuation frames own
   adoption and durable accounting. This adds no optimizer, credit history or
   source parser.
4. The resident verifier retains its existing scope. This predicate is used only
   by the drained recipe handoff; resident polling does not yet load declarations.

The new crisp predicate proves immutable metadata and exact set inclusion; it
makes no model-facing, scheduling, quality or memory-capacity decision.
