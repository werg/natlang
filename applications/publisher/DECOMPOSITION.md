# Publisher: decomposition, part by part

Status: draft for owner review (plans/OWNER_REVIEW.md). Nothing is restructured until the owner has reviewed it.

`publishBrief` (`index.ts:137-144`) reads pinned passages, calls `outline`, calls `compose`, checks the document and
publishes it atomically. The crisp half is right: the check (`index.ts:43-68`), the two renderers
(`index.ts:70-96`) and the three-times-checked symlink switch (`index.ts:99-131`) are exact durability and exact
verifiers. The natural-language half has two problems:

- `Outline` cannot say which table or asset belongs to which section (`types.ts:4` has flat `selected_tables` and
  `selected_assets`), although `outline.nl:11-12` asks for exactly that. `compose` re-derives the assignment.
- `compose.nl` is one call that writes every section, every claim, every table choice and asset choice, and copies
  `evidence_revision` back (`compose.nl:12-18`). A failed check discards the whole document (`index.ts:142-143`).

Decisions: **fn**, **inline**, **implicit**, **crisp**, **service**, **host**, **pluggable**.

## Policy

- **Natural language: planning and writing.** The title and sections, which passages, tables and assets belong to
  each section, each section's prose and claims.
- **Crisp: checking and publishing.** `check` verifies document shape, the evidence revision, tables, assets and
  every claim's span, revision and quote (`index.ts:43-68`). `renderText` escapes and renders both formats from one
  tree (`index.ts:70-96`). `publish` writes, re-verifies and switches a symlink (`index.ts:99-131`).
- **Per-section composition in parallel.** Each section is one call over its own passages and table. The sections
  are independent, so they run at once, and a section that fails its check is the only one retried.
- **The model never copies host values.** `evidence_revision` is the pinned `collection_revision`; claim
  `revision` comes from the passage; `table_id` and `asset_ids` come from the outline's assignment. The assembler
  writes them.
- **Numbers come from sources.** Table cells are printed by the renderer from the table store. A number in prose
  occurs in a cited passage or in the section's table; crisp checks that.
- **State model.** `publishBrief` is decide-then-commit: decisions on a snapshot of the pinned evidence, then a
  pure assembly, then the checked atomic publish. Derived values form a DAG: passages, note, outline, section
  drafts, document, check, rendered files, link.

## Parts

| Part | Decision | Unit | Why |
| --- | --- | --- | --- |
| Pinned read of passages by span id | service | `evidence.read`, `index.ts:139` | Exact, shared with the evidence app. |
| The editorial file the brief names | fn (shared) | `plan/readNote` | Sub-task of `outline.nl:10-11` and `compose.nl:13-14`. Built-in after N1. |
| Title and section list: heading, purpose, passage ids, table id, asset ids | fn | `plan/outline` | One planning task. The assignment is per section. |
| Outline validity: passage ids from the offered set, table and asset ids offered, each used at most once, at least one section | crisp | `plan/checkOutline` | Exact verifier of the stage output. |
| One section: body prose and claims | fn, per section, in parallel | `write/writeSection` | The core task; one section fits a small model's call. |
| Quotes copied from passages | crisp check | `Claim.quote` type | Replaces "The quoted words must occur in the passage" (`compose.nl:15`). |
| Claim `revision` | crisp | `write/fill` | Copied from the passage. |
| Document assembly: title, evidence_revision, sections, assets | crisp | `write/assemble` | Replaces "Carry collection_revision exactly" (`compose.nl:17`). |
| Numbers in prose appear in the cited passages or the section's table | crisp | `write/checkNumbers` | Replaces "never calculate or invent exact table values" (`compose.nl:17-18`); an exact scan. |
| Retry a failing section once with the problem | crisp control | `publishBrief` | Decide on a snapshot, retry once with the problem. |
| Document check | crisp | `check`, `index.ts:43-68` | Exact verifier. |
| Escaping, Markdown and HTML rendering from one tree | crisp | `renderText`, `index.ts:70-96` | Exact. |
| Target name pattern | crisp | `index.ts:100` | Exact. |
| Write both files, verify hashes, re-check twice, switch symlink | host | `publish`, `index.ts:99-131` | Exact durability. |
| Event log | host | `drainEvents`, `index.ts:133` | Mechanism. |

## Natural-language functions, step by step

### `plan/outline`

```
args: brief: string, passages: Passage[], table_ids: string[], asset_ids: string[], note: Untrusted<string>
returns: { title: string,
           sections: { heading: string, purpose: string, passage_ids: string[],
                       table_id: string | null, asset_ids: string[] }[] }
```

1. Read `brief` and `note`; state in one phrase what the document is for.
2. Group the passages by what they say; each group becomes one section. Order the sections so each builds on the
   one before.
3. Give each section a heading and one sentence of purpose.
4. Assign to each section the passage ids it draws on. Assign a table or an asset to the section it supports; use
   only offered ids. Leave `table_id` null when no offered table fits.
5. Write the title.

### `write/writeSection`

```
args: brief, heading, purpose, passages: Passage[], table: Table | null, note: Untrusted<string>
returns: { body: Is<string, "every number in it occurs in a passage or in table">,
           claims: { text: string, span_id: string,
                     quote: Is<string, "occurs literally in the passage with span_id"> }[] }
```

1. Read the passages for this section.
2. Write one claim for each fact the section states: a short claim text, the passage id, and a quote copied from
   that passage.
3. Write the section body from the claims, adapting wording to `brief` and `note`.
4. When the evidence is uncertain or partial, say so in the body.
5. When a `table` is given, refer to its values as the table presents them; the renderer prints the rows.

## Refinement candidates

| Slot | Proposed type | Check |
| --- | --- | --- |
| `Outline.sections[].passage_ids` | `Is<string[], "ids from the offered passages">` | crisp |
| `Outline.sections[].table_id` | `Is<string \| null, "an offered table id used by at most one section, or null">` | crisp |
| `Outline.sections[].asset_ids` | `Is<string[], "offered asset ids">` | crisp |
| `Outline.sections` | `Is<Section[], "every offered passage is assigned to a section">` | crisp |
| `Claim.quote` | `Is<string, "occurs literally in the passage with span_id">` | crisp |
| `Claim.span_id` | `Is<string, "the id of a passage assigned to this section">` | crisp |
| `Section.body` | `Is<string, "every number in it occurs in a cited passage or in the section's table">` | crisp (number scan) |
| `Section.heading` | `Is<string, "equals the outline heading">` | crisp |
| `Document.evidence_revision` | host-filled | crisp |
| `Document.assets` | `Is<string[], "the union of the sections' offered asset ids">` | crisp |
| `Passage.text` | `Untrusted<string>` | crisp marking |
| Body claim wording | `Is<string, "restates its quote without adding facts">` | judged (optional; see question 2) |

## Model-facing changes needing live measurement

1. **New `Outline` shape and `outline` steps.** Measure assignment correctness (every passage placed once, tables and
   assets on the right section) on `ts-host/test/publisher.test.mjs` briefs.
2. **`compose` becomes per-section `writeSection`.** Measure document-check pass rate and cost per document
   (calls grow with sections; each is smaller).
3. **Removed copy obligations** (`evidence_revision`, claim `revision`, table and asset choice).
4. **Number scan replaces "never calculate or invent exact table values".** Measure how often prose contains a
   number absent from sources, before and after.
5. **Retry text** for a failing section (the check's one-sentence detail, for example "quote not found in span X").
6. **"Read only that file"** (`outline.nl:10-11`, `compose.nl:13-14`) becomes the positive `readNote` function.
7. **`Untrusted<string>`** rendering for passage and note text.

## Questions for the owner

1. The final check can fail on a section whose passages changed meanwhile (`index.ts:47`: evidence revision
   changed). That case rejects the whole publication, which is right; confirm no partial publish is wanted.
2. A judged check that a claim "restates its quote without adding facts" would catch claims that overstate. The
   evidence app proposes the same support judgment (evidence question 1); decide both together.
3. `Table` values are only ever printed from the store. Should a section be allowed to ask for a computed column
   (a sum, a percentage) with exact arithmetic done by a crisp helper? New behavior, not in this draft.
