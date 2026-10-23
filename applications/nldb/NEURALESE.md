# Storing and indexing native state

NLDB rows can name exact `nzs1_…` native-state artifacts. Each artifact preserves its source text and tokenizer/model
identity together with token IDs, positions, embeddings, and the available positional KV or hybrid state tensors. The
same archive stores named `nnt1_…` tensor artifacts through the full `NeuraleseStore` API; tensor artifacts are kept
as typed tensors and are never flattened into Neuralese values.

## Folder archive

`FolderBlockStore` writes the native safetensors artifact bytes under `blocks/<artifact-id>`. IDs are content addresses,
so repeated writes are idempotent. Reopening the store rebuilds its metadata index from the files. `pin` protects
in-flight use, and `collect` removes unreferenced artifacts while retaining their exact native dependency closure.
NLDB rows that contain a `neuralese` value hold an `nzs1_…` state ID; keep the row and artifact changes in the same
successful transaction boundary.

## Dialect-aware retrieval

A state is meaningful in the architecture and checkpoint/tokenizer context recorded by its metadata. The semantic IVF
index only indexes the state's learned token embedding tensor, retaining its dialect (`native-state/<architecture>`)
and width. It refuses a query encoded in another dialect or width. Named tensor artifacts remain available by their
`nnt1_…` IDs for exact parameter, gradient, optimizer, or adapter operations; they are not candidates for semantic
nearest-neighbour search.

Pooling token embeddings to one normalized vector loses token order and the other shared state tensors. IVF is a recall
stage, not a substitute for reading or consuming the original native state. Text readers do not reinterpret the native
state: a model must explicitly support native-state consumption and the matching native identity.
