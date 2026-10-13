# Application-scale semantic source contract

This contract is for authored natlang projects whose algorithm lives in a tree of `.nl` callable folders. It does not replace the existing 220-program catalog adapter. A project writes its own `DECOMPOSITION.md` before implementation, mapping every decision to a natural-language function or instruction and justifying every crisp helper.

## Manifest and immutable snapshot

`project.json` names a stable project ID, source root, entry `.nl` file, typed request, task input snapshot, oracle file, source-world split key, comparison specification, provenance, license and any services. `program_files` lists every `.nl` and declaration file included in the source snapshot by relative path. `fixture_files` lists every model-visible task input by relative path. `oracle_path` is outside the model-visible snapshot. The packager hashes the exact manifest, source bytes, input bytes and oracle bytes; a later edit creates a new source revision.

The manifest also declares `typed_call_graph`, `scoped_literals` and `iteration_reducers`. These are audit claims about the project, not a template from which the packager rewrites its functions. The packager checks that declared `.nl` paths exist, every named child is callable from its actual parent folder, referenced types exist, service names are declared, and the model-visible root has the declared request/result signature. It preserves source text and folder structure byte for byte. A dependency outside the snapshot or an absolute/escaping path is an error.

## Execution and admission

The packager emits a Program/2 source with the complete project files and one pinned task input snapshot. Its first exact pass verifies path containment, type/signature availability, fixture/gold separation, provenance and split-world identity. A scripted reference, if supplied, is source/runtime replay evidence only; it cannot stand in for the app algorithm or establish answer quality. Teacher execution must actually call the authored root, traverse the relevant named branches, consume dynamic literal results, advance structural iterations and produce the typed result.

Answer comparison preserves explicitly ordered sequences, declared unordered multisets and exact IDs/numbers/booleans. Prose alternatives and alternate valid plans receive independent source-grounded review; a generic judge cannot override changed exact bindings. Stage and final-output review checks source-path/quote witnesses, distinct semantic contributions, actual child consumption, iteration progress, branch coverage and the project's exact verifier. Only a complete reviewed native trajectory enters the training corpus. Failed or interrupted diagnostics are not model-negative or DPO examples unless an assigned recovery consumer and correct preference pair are recorded.

Source size and semantic depth are mix labels, not admission limits. Valid easier projects and the earlier 220 cases remain eligible. The new slate should use application-scale organization because its tasks require it; no arbitrary file-count or call-depth threshold is encoded as a language rule.
