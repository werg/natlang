# JSON file quality continuation

Policy: preserve the original teacher execution and independent gold. When a
TRAIN string-map task explicitly requests a JSON object, equivalent whitespace
and object-key order do not constitute a failed answer. An unrelated byte
contract, wrong answer, modified input file or other failed check remains failed.

1. Read the exact original captured row and its source file binding.
2. Verify the declared JSON string-map contract, independent returned answer,
   and every original check except the sole file-format failure.
3. Invoke the existing `checkFiles` JSON-string-record comparator on the full
   actual, expected and input files. Crisp code verifies this exact contract;
   it does not choose a new answer or introduce an alternate comparator.
4. Produce a separate evaluation view, preserving program, trajectory, actor,
   expected values and original bytes. Bind its exact original and verifier.
5. Conversion reconstructs this view from the original and re-runs the verifier;
   source/packet/protection and per-action admission checks remain mandatory.

The view grants neither student exposure nor an on-policy claim. Future task
source derivatives may correct only this file-oracle metadata; their original
model-facing program and independent reference remain unchanged.

Historical verifier identity is strictly typed provenance bound by the exact
source artifact. It is not admission authority: a current canonical comparator
must reconstruct the evaluation from the original, and every other field must
match exactly. A changed comparator rejection invalidates the continuation;
an unrelated runtime rebuild with the same result does not erase valid custody.
