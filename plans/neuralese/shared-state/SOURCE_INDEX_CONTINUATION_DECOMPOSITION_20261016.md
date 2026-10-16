# Continue source validation across the internal prefix extension

Policy: reuse a completed source-validation decision only when its input and
validation semantics are unchanged. The new prefix optimization acts after
source parsing. It does not admit data, earn exposure, certify capacity or
qualify numerical training behavior.

1. Compare the seven files defining the existing native source-index policy.
   Six must be byte-identical. The compiler may differ only by the explicit
   default-false internal prefix field, its stable serialization helper, and
   three guards unreachable for ordinary parsed invocations.
2. Verify that the unchanged parser never sets that field. Remove only those
   exact additions from the new compiler AST and require the entire remaining
   AST to equal the old compiler. Any other change requires normal validation.
3. For explicitly requested source hashes, verify the existing index's complete
   key, filename and metadata digest. Preserve every entry and every other key
   field. Write an atomic cache entry under the new policy hash and verify it.
   Missing entries remain cache misses; conflicting existing entries fail.
4. Record old/new policy-file identities and old/new cache content hashes.
   The ordinary loader still checks source content, tokenizer, split, roles and
   metadata. No trainer, admission, held split, optimizer or visit state changes.

These decisions are stated here in natural language. Crisp Python performs AST
comparison, exact hashing, cache-key verification and atomic writes using the
existing row-index mechanism. It does not create an alternative validator.
