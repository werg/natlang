# Neuralese files (`.nz`)

Version `natlang.neuralese-file/1`, 2026-10-03. Normative. Part of the Neuralese
section of [SPEC.md](SPEC.md).

A `.nz` file stores typed named exports whose values may mix exact data and soft
values, together with every native artifact those values need. Soft functions,
data blocks, native adapter coefficients and the standard library's operator
bodies are all values in such files.

## Container

A `.nz` file is a [safetensors](https://github.com/huggingface/safetensors)
file.

- **Tensors.** One `U8` tensor per distinct native artifact, named by its
  content ID: a native state (`nzs1_…`) or a named tensor artifact (`nnt1_…`).
  The tensor holds the artifact's own exact encoding (a safetensors body with
  its metadata and all of its tensors), so the ID can be checked against the
  bytes. An artifact used by several exports is stored once. Tensors are
  ordered by ID.
- **Header.** The safetensors metadata holds one key, `natlang`, whose value is a
  JSON document conforming to
  [neuralese-file.schema.json](neuralese-file.schema.json).

Standard safetensors tooling can open the file.

## Header

| Field | Meaning |
| --- | --- |
| `format` | `"natlang.neuralese-file/1"`. |
| `dialect` | The default dialect tag of the file's values, e.g. `"native-state/lfm2"`. |
| `types` | Natlang type declarations (`type X = …;`) available to the export types. |
| `exports` | Map from export name to `{ type, value, description? }`. `type` is in natlang type syntax. |
| `native_states` | Map from `nzs1_` ID to `{ bytes }`, one per native state tensor. |
| `native_tensors` | Map from `nnt1_` ID to `{ bytes }`, one per named tensor artifact (adapter coefficients and bases, gradients, optimizer state, derivative programs). |
| `skill` | Optional. `name`, `description`, `body` (the export that holds the skill body) and an optional `natlang` block. |
| `provenance` | Optional. Producer, source, and parent file revisions. Native source captures named here are dependencies and are stored in the file. |

A header with a flat `blocks` inventory (the retired projected-vector format) is
rejected (`neuralese-file-format`).

## Values

An export value is JSON with these tagged forms:

| Form | Meaning |
| --- | --- |
| `{ "$neuralese": { "type": "Neuralese<T>", "id": "nzs1_…", "stop_gradient"?: true, "source_occurrence"?: "…" } }` | A soft value: a native state of this file. |
| `{ "$neuralese-fn": { "type": "Neuralese<F>", "body": "nzs1_…", "body_stop_gradient"?: true, "body_source_occurrence"?: "…", "captures": { name: value } } }` | A soft function: its body state and its snapshot captures. |
| `{ "$nativeAdapter": { "id": "nnt1_…" } }` | A native adapter coefficient artifact of this file. |
| `{ "$nativeTensor": { "id": "nnt1_…", "kind": "…" } }` | Another named tensor artifact of this file; `kind` must match the artifact's. |
| `{ "$ref": "export" }` or `{ "$ref": "./other.nz#export" }` | Another export of this file or of another `.nz` file. |

Everything else is ordinary JSON data. A function with live captures cannot be
saved (`neuralese-live-capture-save`).

## Loading

- Loading checks every export against its declared type, as call boundaries do
  (`neuralese-file-type`), and the dialect of every referenced state against the
  export type's dialect when the type names one (`neuralese-dialect-mismatch`).
  A native state's dialect is `native-state/<architecture>`, or
  `semantic-state/<adapter SHA-256>` for a semantic state.
- An artifact whose ID does not match its content is rejected
  (`neuralese-block-integrity`).
- Every dependency of a stored artifact (its adapter coefficients, the parent,
  gradient and optimizer state of an optimized state, the inputs of a
  derivative program) must be in the file (`neuralese-unknown-block`).
  Optimized-state ancestry is verified and must form a DAG.
- `$ref` cycles are rejected (`neuralese-ref-cycle`); references form a DAG.
- Application and callable-folder code import exports by name:
  `import { rubric, triage } from './support.nz'`. The build generates
  `support.d.nz.ts` from the header, as `.nl` files get `foo.d.nl.ts`.
- In a callable folder, a `.nz` file is a context item: a data entry named after
  the file (without `.nz`) whose value holds its exports. Soft function exports
  are function-typed data whose calls go only to their definition site's
  context.

## Skills

The header may carry a `skill` field: `name`, `description`, `body` naming an
export, and the optional `natlang:` block of a standard `SKILL.md` frontmatter
(`scope`, `exports`, `requires`, `tests`, `provenance`). Loading checks that the
field has a name, a description and a body that names an export. The skill
loader does not read `.nz` files: skills are `SKILL.md` folders, and a skill
refers to native state through typed values in its metadata.

## Content and revisions

A file's content is a function of its header and artifacts. Changing a value
produces new file content; revision history is the history of the containing
context. Writers produce canonical JSON (sorted keys) so equal content gives
equal bytes. A writer stores the complete dependency closure of every value it
saves and checks that the result loads before writing it.
