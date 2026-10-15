# Native graph data conversion and production recipe binding

This is the data-side implementation plan for the shared-state migration. It
preserves source trajectories and emits truthful native graph supervision for
both backbones; it does not infer edges from legacy marker strings or assume
that token IDs transfer between tokenizers.

## Natural-language stages

1. Read admitted source rows, exact source manifests, split/source-group
   bindings, whole-program exclusions, decision holds, and omission ledgers.
   Keep train and held rows distinct. A changed proof invalidates only the
   dependent conversion frontier and is rechecked before admission.
2. Recover each invocation's complete context and target from its attested
   source record. Preserve system/user/tool/assistant roles, actual tool output,
   full turn order, exact supervised target boundaries, and source tokenizer
   provenance. Represent ordinary text as one graph invocation.
3. For authenticated producer/consumer rows, join the exact source record,
   typed write witness, payload piece, reader occurrence, and declared edge.
   Verify source IDs, body hashes, split/group closure, and actual payload text.
   Emit structural producer/consumer markers and import spans. Keep paired
   model-neutral source/twin IDs and producer `{invocation,start,stop,text,
   source_span_id,text_sha256}` alignment metadata. Use exact token spans only
   after encoding with the destination model's expanded tokenizer.
4. When a source provides a typed write without an authenticated reader, train
   its producer and complete ordinary trajectory as an explicit one-node or
   multi-node static-transformation contribution. Do not fabricate a native
   edge. If source text or witness is missing, write a repair/omission row with
   the precise reason; do not silently discard the source.
5. Validate each graph with the shared graph parser and exact verifier:
   tokenizer fingerprint, node/token masks, source/split closure, acyclic
   imports, producer-span membership, text/payload hashes, and whole-chain
   train/held disjointness. Verify role markers and targets correspond to the
   source witness, not a rendered guess.
6. Stream compact graph parts and manifests for LFM and Mellum using the same
   converter and policy. Reuse tokenizer-independent source indexing and
   payload records; model-specific tokenization is an explicit expansion
   keyed by each tokenizer fingerprint. Keep input source pins and every
   omitted/repair row in the derived manifest.
7. Bind both model recipes to the admitted full text/native source union,
   protected held graphs, task packet, and adaptive continuation data. Stages
   consume actual graph rows and report unique source, trajectory, token, and
   producer/consumer exposure; admission or registration alone is not exposure.
8. Once two model native prerequisites complete, the executable shared recipe
   schedules resumable joint semantic-space adaptation, adjustable write-out,
   and behavioral distillation using paired source/twin alignments. This data
   conversion supplies the aligned source rows but does not claim a trained
   VAE or learned cross-tokenizer interchange.

## Crisp mechanisms and reason

- Source readers, compression, hashing, tokenizer encoding, graph parsing,
  split checks, import offsets, manifests, atomic publication, and recipe input
  validation are crisp: they are exact data plumbing or exact verifiers.
- The shared graph objective and model-specific tokenizer expansion are the
  existing native trainer interfaces. Do not create per-backbone converters or
  another renderer/sampler.
- Source-specific semantic mapping remains a declared natlang transformation
  policy where it judges ambiguous content. This implementation only uses
  exact structured witnesses and authenticated source references for edges.
- Every unsupported legacy/native case remains an explicit ordinary-text
  contribution or a repair/omission entry with provenance, never a guessed
  state edge.

## Validation required before admission

- Small fixtures for ordinary full-context text, one exact producer/consumer
  edge, multiple readers, held/train overlap, missing source witness, tokenizer
  mismatch, payload hash mismatch, and repaired versus genuinely missing source
  context.
- Source-specific streaming counts equal pinned input counts plus explicit
  exclusions/omissions; every retained trajectory is represented once in its
  source lineage and every aligned pair has one matching model-neutral twin ID.
- Both model recipe plans resolve the exact admitted graph and text manifests,
  held packets, and task packet. A focused trainer execution verifies actual
  graph exposure and continuation; no stage-name-only completion qualifies.
