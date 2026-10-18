# Compact controls for an exact full native parent

The operational policy is to keep the full optimizer/RNG/exposure checkpoint as
training authority while avoiding repeated unpickling of its large Python
training metadata for input staging and inference. Compact controls are an
explicit, immutable derivative of the same already-verified saved object; they
do not replace training state or grant readiness, completion, or exposure.

1. During an already-required full-state decode, use the existing dependency
   verifier with that same in-process saved object to enumerate consumed input
   bindings. Keep source, base, objective, pending, and historical source rules.
2. Seal the exact source SHA/bytes, model identity/configuration, saved history
   profile, resumability checks and exact dependency closure beside the existing
   bitwise-verified model-only actor view. Bind the executing producer mechanism.
3. A provider may consume only an explicitly hash-bound compact control receipt
   for that exact full source and supported native leaf form. Verify every
   declared dependency through existing namespace and asset checks. Default and
   unsupported forms retain full checkpoint verification.
4. Raw actor preparation verifies the same exact source and actor view, restores
   the saved tokenizer profile, and derives identity from actual view tensors.
   The full training learner still restores its complete parent exactly once.
5. Missing optional controls use the strict existing path. Explicit mismatches
   reject; they never invent optimizer/RNG state, change actor identity, or hide
   corrupted artifacts. Training continues if an optional producer fails.

These are crisp serialization, provenance, filesystem verification, and process
plumbing. No prompt, mathematical operation, sampling policy, source admission,
source credit, or numerical qualification rule is changed. Both students use
one mechanism. Independent review and actual closure/identity equality tests
precede operational adoption.

Publication uses an explicitly versioned immutable directory. Repeating the same
source under the same version is idempotent; a changed producer or adaptation
uses a new versioned root, retaining the prior bound controls for sealed work.
The mutable choice of root belongs to the verified future continuation. Each
bridge request resolves it once into immutable config bytes; absence at that
point freezes the full path even if publication completes later.

The exporter specification binds both producer files by SHA and byte length and
executes those bytes in private namespaces. It retains the same saved object
through actor-view readback and control derivation. Optional errors preserve the
full pin/export and record the typed repair. On old qualified packages the new
helper calls the existing uncached full hasher when the strong cached API is
absent; it never substitutes archive receipts or changes the source-index
hashing module. All file checks retain device/inode/size/mtime/ctime guards.

Transport enumerates source/view/control files without deserializing model
weights. It authenticates the actual consumed full-parent alias by SHA/bytes;
the producer's old physical pin path is provenance and is not an additional
48GB transfer dependency. Raw actor loading still verifies the actual compact
tensor payload against its receipt. Learner loading stays full and resumable.
Dependency verifier revision, namespace or read-only-root differences fall back
to the original full traversal. Explicit corrupted bindings reject.

For deployed old packages, the reviewed bootstrap can explicitly install a
hash-bound copy of the already-published strong hashing helper in a private
module. The original package module is untouched. This selection comes from the
exporter/control launch specification, never from serialized checkpoint state;
its file binding is retained in the control dependency closure. Repeated control
reads reuse that helper's existing device/inode/size/mtime/ctime content cache.

The exporter retains the five-field witness across the actual full decode,
checks it before its owned rename, and retains the mapped pin witness before
optional actor-view extraction. The control producer receives this retained
witness. Mutation during decode or between view extraction and the control hook
rejects, including same-size rewrites with restored mtime.
