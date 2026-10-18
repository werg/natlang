# Reviewed source-child oracle coverage

Policy is an explicit independent natural-language review: for each exact
captured authored function, decide whether its child asks the same proposition
as the original source's declared expected/announced-failure decision. Read the
complete criterion and source policy, not a keyword or a model answer. A
negated, stronger, unrelated or ambiguous criterion remains uncovered. This is
an offline oracle repair; it changes no generation prompts or recorded output.

1. Bind the original raw TRAIN row, captured module, source tests and oracle.
2. Bind an independent review of exact literal child instructions. Support literal criteria and the explicitly reviewed `${policy}` interpolation
   only when it resolves to one unchanged, unshadowed module const. Support the
   reviewed boolean and above-threshold hold/rollback result contracts; never
   infer arbitrary interpolation or a proceed/below-threshold answer.
3. Derive boolean labels only from original above-threshold source records
   whose original oracle and expected whole result agree: hold means these
   failures were announced; rollback means they were not. Full note text must
   have one consistent label. Below-threshold proceed is not boolean evidence.
4. Match both the reviewed exact child criterion and full source note text in
   the actual opening. Use the existing structural result contract. Reject
   conflicting matches; an uncovered child still fails.
5. Reproduce the historical coverage failure, then execute every unchanged
   original test through the current compiler and approved source aliases.
   Require exact whole outputs and every other captured check before deriving
   a separate quality view. Original files, source tests, trajectory, actor,
   runtime provenance and raw outcome remain immutable.

Review is the semantic policy. AST/source hashing, typed canonical equality,
source-label joins, unchanged executable tests and durable proof bindings are
crisp verification. Existing argument-alias and quality-continuation mechanisms
carry the result; no converter or per-model implementation is added.
