# Native serving adapters

## Natural-language service policy

1. A caller creates an Adapter for the exact live native model, chooses its
   declared layer matrices and rank, and tunes its coefficients against actual
   native graph objectives. Nested calls combine explicitly scaled adapters.
2. Apply the same adapter to every prefill, cached step and decision score in
   that call. Leave other calls and the underlying model unchanged.
3. Persist the immutable coefficients, exact frozen bases, model identity and
   optimizer continuation together. Reject another model's adapter rather than
   ignoring it or applying a legacy projected channel.
4. Exported state names the effective adapted model. An unadapted consumer may
   explicitly reencode its text; it may not splice adapted KV as base KV.
   An existing learned semantic decoder is qualified only for its bound model.
5. Unsupported higher-order objectives and adapted-policy RL replay remain
   explicit errors until their actual execution contracts are implemented.
   Direct adapter support is not a claim of those separate capabilities.

## Exact mechanisms

The existing AdapterBank coefficient algebra and SVD sign convention remain
one implementation. Native bindings select actual named backbone matrices,
including batched expert matrices. Frozen U/V bases (and tiny projections) are
stored as named native tensor artifacts, with small coefficient artifacts
referencing their exact basis artifact. No projected communication vectors are
introduced. The tensor store's codec, SHA identity, owner holds and restore
transport provide durability.

A shared adapter execution scope substitutes only selected weight tensors for
base-plus-low-rank deltas. It holds a model-wide reentrant lock until all
forward/backward checkpoint recomputation finishes and restores original
parameter objects on every exit. Unadapted native forwards use that same lock.
This exact mechanism handles both ports' functional projections and expert
weight accesses without per-model hooks or changes to unadapted arithmetic.
It materializes the selected effective weight matrices, so users explicitly
choose targets; this is a declared memory cost, not a zero-copy LoRA claim.

The serving API preserves the existing adapters request list. Native create,
graph coefficient gradients and immutable optimizer updates use typed nnt1
artifacts and the existing first-order gradient/optimizer routes. Effective
checkpoint identity hashes the ordered exact adapter IDs and scales alongside
the live base identity. A state receipt retains that binding for durable replay
and honest cross-model import classification.

## Qualification and open work

Support remains unadvertised while implemented. Required evidence: actual LFM
and Mellum cached/graph outputs and coefficient gradients; explicit weight
oracle; nonzero coefficient updates with optimizer continuation; restoration
including exceptions/concurrent unadapted calls; exact codec/model/basis
mismatch rejection; real HTTP create/score/generate/grad/update/restore path.
Adapted-policy RL and higher-order objectives must not inherit support from
first-order graph learning. Production arithmetic and training jobs are not
changed by this work.

## Implemented internal path and review evidence

Native coefficient/basis/gradient artifacts, functional parameter and frozen
expert-buffer substitution, generation, decisions/batched scoring, forced
writes/body initialization, first-order graph learning and immutable Adam/SGD
continuation now use this shared path. Native state carries the exact adapter
binding with canonical hexadecimal scale strings. `.nz`, compiled imports,
Node/File/Memory archives, executable dependency retention and restart restore
retain the bases, coefficients and optimizer state. OPFS lacks named-tensor
storage and explicitly rejects these files; it does not silently lose them.

A populated Mellum deployed expert cache could otherwise ignore substituted
expert weights or leak them back into a base call. Only the actual substituted
bank bypasses that base cache; a regression checks warm base → adapted expert
→ bitwise base plus unchanged cached storage pointers. Original parameter and
buffer objects are restored, including ties and exception exits. The scope
covers backward/checkpoint recomputation; higher-order/donated-buffer replay is
not claimed.

Native tensor transport now retains the existing identity-bearing canonical
metadata bytes from wire across numeric spellings such as Python `1.0`/`1e-09`
and JavaScript `1`/`1e-9`. It changes no existing hash domain, source manifest,
checkpoint or recipe identity. An actual Python → JS durable archive → Python
fixture covers integral/exponent values, negative zero and nested stats; field
mutation with preserved old metadata is rejected.

Ledger evidence (2026-10-16):

- `native-serving-adapters-final2-cpu-20261016`:32 passed,222.19s,
  including actual backbone optimizer/RNG relocation continuation and native
  artifact/store regressions.
- `native-serving-adapters-cuda2-20261016`:3 passed,15.51s, actual LFM and
  Mellum producer/consumer graph/reference hidden/logit/coefficient adjoints at
  unchanged atol3e-6/rtol2e-5, plus actual BF16 Mellum expert-bank gradients.
  This is local Torch2.11/CUDA qualification, not a cloud throughput claim.
- `native-serving-adapters-node-regression-20261016`:19 existing Node
  codec/state/value/file-store cases passed.
- `native-serving-adapters-graph-regression-20261016`:5 passed, existing native
  gradient fixtures, shared legacy coefficient/SVD algebra and deployed cache.
- `native-serving-adapters-import-20261016`:2 passed, actual adapted state
  imports with coefficient and KV gradients; missing exact adapter rejects
  the foreign effective state.
- Final `native-serving-adapters-finish-20261016`:build passed,4 actual typed
  HTTP/cache/tie cases passed12.49s, quick passed. HTTP exercises both ports'
  coefficient learning, multi-token scoring, generation, raw writes, portable
  state/basis/optimizer archive and remote restart restore.

The initial combined regression included the legacy adapter suite's full-size
LFM fixture and exceeded a3GB claim: that unit is OOM/INCOMPLETE, not green.
The rerun excludes that expensive fixture; direct shared legacy algebra and the
standing full suite cover its changed helper.

Direct native adapter capabilities remain **unadvertised pending independent
review**. Adapted-policy RL, higher order, legacy projection decoders and GGUF
LoRA export remain separate explicit unsupported paths. This is not full
learning-continuum or production cutover completion. The existing production
models, recipes and unadapted kernel arithmetic are unchanged.
