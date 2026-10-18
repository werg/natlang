# Reuse verified capacity source trees

The recipe chooses complete graph cases and their exact source bindings. This
change does not choose cases or alter their geometry, losses, tolerances or
admission. It makes the existing capacity verifier reuse data it has already
validated for the running trainer.

1. Keep the existing request identity, index-witness and source-binding checks.
2. Seed the existing `NativeSourcePreparation` mechanism from the loaded train
   tree. It retains exact tokenizer, roles, split, model-view and file identity.
3. Load the requested source through that mechanism. Existing roots or children
   are reverified before reuse; other sources use the unchanged parser. Keep
   each preparation local to one check so unrelated requests do not retain an
   unbounded second collection of source indexes.
4. Run the existing numerical-reference and whole-graph backward checks and
   preserve their original results, state restoration and certificate rules.

The recipe and request are the natural-language policy boundary. Steps 1–3 are
crisp identity and storage plumbing, using the same implementation as resident
append preparation. Step 4 is the existing exact engineering verifier. No
model-facing text, parser component, model arithmetic or live process changes.

Validation uses both model families through the actual prepared-capacity route,
with real graph sources and backward execution. Additional cases reject changed
leaf data and wrong role/split contracts before model work. There is no cloud
cutover or capacity credit from these CPU tests.
