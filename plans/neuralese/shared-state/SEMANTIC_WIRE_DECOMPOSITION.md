# Persisted shared semantic dialect

The native writer produces ordinary text, source token embeddings and raw
positional backbone state. Once a checkpoint-bound semantic adapter exists,
encode its state into the shared per-token posterior mean. Store text, actual
source token IDs, positions, embeddings, character offsets, semantic codes and
adapter identity through the existing `nzs1_` durable artifact lifecycle.
Raw KV is not part of this compact wire representation.

1. The writer verifies the source tokenizer and adapter/model identities and
   emits the semantic message. Its raw span remains available in memory.
2. A consumer verifies the exact adapter and source identity, aligns retained
   character offsets to its tokenizer, prefills isolated target text, decodes
   target-state deltas and splices the resulting native block. Same-model reads
   also decode semantic messages; a code tensor is never treated as raw KV.
3. Without the matching adapter, ordinary text remains an explicitly reported
   best-effort transport. Corrupt artifacts fail their existing content hash.
4. RL capture archives the actual decoded target block as its observation. For
   producer replay it additionally archives the raw source block, binds that
   archive in the write receipt, and validates both representations' source
   token/embedding identities. Only RL capture retains this extra raw artifact.
5. Python and TypeScript share the same schema, tensor inventory, content hash,
   storage/holds and archive/restore mechanisms. No parallel artifact store.

Encoding, alignment, exact identity verification and durable tensor transport
are crisp mechanisms. The recipe declares when to train and deploy an adapter;
the supervisor chooses improvements using observed task behavior and throughput.
Complete live serving, cross-language persistence and RL replay tests precede
claiming this as the shared dialect replacement.
