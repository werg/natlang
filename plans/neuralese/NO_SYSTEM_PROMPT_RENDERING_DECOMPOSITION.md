# Program-header rendering (no generic system prompt): decomposition

Owner decision 2026-10-11: the Neuralese students (Mellum2.1 and LFM2.5) are fine-tuned only for natlang, and
generality beyond natlang does not matter. Training has a warm-up phase on documents with the full runtime system
prompt. After it, documents drop the GENERIC runtime instructions, which distils the prompt into the weights.
PROGRAM-SPECIFIC content always stays in the documents (owner, confirmed 2026-10-11).

The serving renderer must match training. A model trained past the warm-up must be called with the same
program-header rendering. This document plans that change for both machines. The data side is implemented;
see section 5.

## 1. Measured composition (4962 union, LFM2.5 tokens, mean per document)

| Part | Tokens (mean / median) | Class |
| --- | --- | --- |
| Generic runtime prose (interpreter prompt, approach, cut-off, rejection guidance, nl/iterateOn docs) | about 2,260 / 2,160 | generic: drop after warm-up |
| Mode tool docs and schemas (eval, read_code, read_page, compact_history; folder file tools; edit_code) | about 1,840 | generic per runtime mode: drop |
| `return_result` schema with the typed `value` | 268 / 277 | program-derived, but redundant: the signature and type aliases are in the user turn |
| Call-property paragraphs: directory-reducer declaration, nesting layer, Neuralese parameter/result/emulation declarations, edit_code capability | 0 to about 1,100 (present in 3,233 of 4,962 documents) | **program-specific: keep verbatim** |
| First user turn: `You are inside this call: f(...)`, types, instructions, inputs, skills | 169 / 133 | **program-specific: keep** |
| Assistant, tool and history turns | 789 / 628 | trajectory: keep |
| **Whole document** | **5,335 / 4,454** | |
| **Program-header document** | **1,211** (train 4.14M tokens instead of 17.4M) | |

On the Mellum twin, train tokens fall from 17.6M to 4.28M. The supervised suffix is unchanged: 250,020 train tokens.

The system prompt has 349 distinct variants across the 4962 documents. They are runtime versions times call modes,
not programs: 1,688,424 distinct tokens against 21.5M rendered.

## 2. Units

| Unit | Decision | Why |
| --- | --- | --- |
| Piece classification (`generic` or `program`) | Crisp data on the existing piece registry: add a `scope` field to each `PromptPiece` in `ts-host/src/native/system-prompts.ts` | One implementation per concept. The piece registry already names every runtime-owned text and is what the softening bank substitutes. Program-header rendering is the degenerate bank in which generic pieces map to nothing. |
| Call-property paragraphs (directory reducer, nesting layer, Neuralese declarations, edit_code) | `scope: 'program'` pieces. They are emitted verbatim in a short system turn, the program header | They state properties of this call that the user turn does not carry. For example, only 157 of the 2,770 folder-mode documents mention the folder in the user turn. |
| Tool schemas (`tools` passed to the chat template) | Omitted under program-header. Tool calls are still parsed by name, as today | The model learns the tool surface during warm-up. Removing the schemas keeps the template from re-emitting a generic block. |
| Rendering-mode policy (which mode a model gets) | Crisp setting per model profile: `ModelProfile.promptRendering: 'full' \| 'program-header'`, default `full` | This is mechanism, not a decision made at run time. The mode is a fact about the weights, recorded with the checkpoint (`direct-certificate.json`, field `prompt_rendering`). |
| Python renderer (`natlang_neuralese/serve/chat.py` `render_messages`, `data/text_corpus.native_gold_packet`) | One `prompt_rendering` argument; `program-header` removes generic paragraphs and the tool list from the system message, using the same scope table exported from ts-host (`promptPieces()` JSON) | The renderer used for training data and the serve path must be the same function. The table must not be copied by hand. |
| Engine and serve (`natlang_neuralese/serve/engine`, llama.cpp fork) | No change. They render whatever messages they receive | Only the message content changes. |
| E2E stage (`train/trajectories.py`) | Renders rollouts through the same renderer with the stage's declared `prompt_rendering` | Training rollouts must equal serving rollouts. |
| Recipe | Two phases of `neuralese_direct` (and later E2E): `direct_sft` on the full-prompt binding, then `direct_sft_program_header` on the program-header twin, which `requires` the first | Warm-up then distillation, declared in the shared base so both lines get it. |
| Checks | A crisp verifier compares the ts-host rendering of a recorded call under `program-header` against the Python training rendering, byte for byte | Training and serving parity is an exactness property. |

## 3. Steps (both machines)

1. ts-host: add `scope` to `PromptPiece`. Mark `directory-reducer/*` declarations, `depth-limit-notice`, the
   Neuralese declaration pieces and the edit_code paragraph as `program`; everything else is `generic`. Export the
   table with `promptPieces()`.
   - Split `directoryReducerPrompt` so that its one-sentence declaration ("This call is a directory reducer: ...")
     is its own piece. The file-tool documentation stays a separate generic piece.
2. ts-host: implement `ModelProfile.promptRendering`. When it is `program-header`, the native prompt builder
   (`native/prompt.ts`) emits only `program` pieces and passes no tool schemas. Tool-call parsing is unchanged.
3. Python: add `prompt_rendering` to `render_messages` and `native_gold_packet`. Under `program-header`, keep only
   the system paragraphs listed as `program` in the exported table and pass `tools=None`.
   - Until this lands, `scripts/neuralese_data/strip_generic_system_prompt.py` produces the same documents from
     rendered text. Its `PROGRAM_SPECIFIC_HEADS` list is the interim copy of the table, and it is replaced by the
     export in this step.
4. Verifier: render about 200 recorded calls both ways (ts-host serving and Python training) and require equal bytes.
   Run it in `check-main.sh`.
5. Recipe: declare the two-phase direct stage in `neuralese-direct-v2`'s successor, and the same for E2E. Record
   `prompt_rendering` in the stage certificate.
6. Measurement (debt row in `plans/MODEL_FACING_CHANGES.md`): on the trained student, about 48 held natlang calls
   per variant (full against program-header). Measure first-turn tool validity, finish rate and held CE. The base
   executor is not a comparison point.
7. Skills: update `skills/natlang-integration` with the model-profile setting and the requirement that a model
   trained past warm-up is served with `program-header`. There is no new model-facing name.

## 4. What does not change

- The user turn: signature, type aliases, instructions, input listing, skills.
- Tool-call syntax and the tool names the model emits.
- Error and feedback texts.
- Teacher and executor models that were not fine-tuned this way keep `full`.

## 5. Data side (done 2026-10-11, DGX)

- Script: `scripts/neuralese_data/strip_generic_system_prompt.py`, commit 5397f4f6. It splices at the token level,
  keeps every source ID after the system turn, and re-tokenizes to prove 4,289 rows exactly. The other 673 rows
  contain escaped special spellings and are token splices.
- Registered twins, same ids, splits and source groups, 0 omissions:
  - `neuralese-foundation-text-input-4962-program-header-lfm25-20261011-v1`
  - `neuralese-foundation-text-input-4962-program-header-mellum21-20261011-v1`
- Every new text set (sections C and E of `DATA_INVENTORY_20261011.md`) gets a full-prompt render for the warm-up
  share and a program-header render for the rest. The program-header render is about 4× cheaper per document.
