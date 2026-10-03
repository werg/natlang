# Neuralese files (`.nz`)

Version `natlang.neuralese-file/1`, 2026-10-03. Normative. Part of the Neuralese
section of [SPEC.md](SPEC.md).

A `.nz` file stores typed named exports whose values may mix exact data and soft
values. Soft functions, skills, data blocks and the standard library's operator
bodies are all values in such files.

## Container

A `.nz` file is a [safetensors](https://github.com/huggingface/safetensors)
file.

- **Tensors.** One tensor per distinct soft block, named by its content ID
  (`nz1_…`), with shape `[length, width]` and the dialect's dtype. A block used
  by several exports is stored once.
- **Header.** The safetensors metadata holds one key, `natlang`, whose value is a
  JSON document conforming to
  [neuralese-file.schema.json](neuralese-file.schema.json).

Standard safetensors tooling can open the file. `natlang nz show file.nz` prints
the header with types and references.

## Header

| Field | Meaning |
| --- | --- |
| `format` | `"natlang.neuralese-file/1"`. |
| `dialect` | The default dialect tag of the file's blocks, e.g. `"nd:natlang@1"`. |
| `types` | Natlang type declarations (`type X = …;`) available to the export types. |
| `exports` | Map from export name to `{ type, value }`. `type` is in natlang type syntax. |
| `blocks` | Map from content ID to `{ dialect, length, width, dtype, producer?, truncated? }`, one per tensor. |
| `skill` | Optional. Skill frontmatter (`name`, `description`, and the `natlang:` block) when the file is a skill. |
| `provenance` | Optional. Producer, source, and parent file revisions. |

## Values

An export value is JSON with three tagged forms:

| Form | Meaning |
| --- | --- |
| `{ "$neuralese": { "type": "Neuralese<T>", "id": "nz1_…" } }` | A soft value: a block of this file. |
| `{ "$neuralese-fn": { "type": "Neuralese<F>", "body": "nz1_…", "captures": { name: value } } }` | A soft function: its body block and its snapshot captures. |
| `{ "$ref": "export" }` or `{ "$ref": "./other.nz#export" }` | Another export of this file or of another `.nz` file. |

Everything else is ordinary JSON data. A function with live captures cannot be
saved (`neuralese-live-capture-save`).

## Loading

- Loading checks every export against its declared type, as call boundaries do,
  and every referenced block's dialect against the export type's dialect
  (`neuralese-dialect-mismatch`).
- A block whose ID does not match its content is rejected
  (`neuralese-block-integrity`).
- `$ref` cycles are rejected; references form a DAG.
- Application and callable-folder code import exports by name:
  `import { rubric, triage } from './support.nz'`. The build generates
  `support.d.nz.ts` from the header, as `.nl` files get `foo.d.nl.ts`.
- In a callable folder, a `.nz` file is a context item. Its exports are data
  entries; soft function exports are function-typed data whose calls go only to
  their definition site's context.

## Skills

A `.nz` file is a skill when its header has `skill`. The fields are those of a
standard `SKILL.md` frontmatter (`name`, `description`) plus the optional
`natlang:` block (`scope`, `exports`, `requires`, `tests`, `provenance`). The
skill's body is the export named by `skill.body`; `scope` names exports that are
injected into the eval scope of a function that has the skill bound.

## Content and revisions

A file's content is a function of its header and tensors. Changing a value
produces new file content; revision history is the history of the containing
context. Writers produce canonical JSON (sorted keys) so equal content gives
equal bytes.

## Distributional blocks

A `.nz` export may store a block as a distribution: a mean tensor and a
per-dimension log-scale tensor, in place of a single payload. Reading it at
Neuralese temperature `τ` delivers `μ + τ·σ⊙ε`; at `τ = 0` it delivers `μ`. The
header marks such exports (`"distribution": "gaussian-diag"`). Training a stored
encoding can then use sampling-based estimators (score-function or
evolution-strategy style updates through `logLikelihood`) as well as
reparameterised gradients, which matters where the execution between the block
and the reward includes discrete choices.

