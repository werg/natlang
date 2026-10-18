# Bounded-memory resident entry digests

## Problem

Resident admission authenticates the full ordered row-entry sequence by hashing
its canonical JSON bytes. The current implementation builds one UTF-8 byte
string for the entire list. At the production LFM prefix size, the entries
remain resident and that additional serialization allocation can exceed the
consumer's address-space limit.

## Design

1. Keep the existing canonical JSON serializer and digest unchanged for normal
   values and small bindings.
2. Add one shared helper that accepts only a list or tuple, feeds `[` and `]`
   plus each entry's existing canonical JSON bytes and separators directly to
   SHA-256, and returns the same digest as canonical serialization of the whole
   sequence. Memory use is bounded by the largest entry serialization.
3. Use this helper only where resident preparation, cache-witness verification,
   and resident polling hash a complete ordered row-entry sequence. Do not alter
   row contents, ordering, cache keys, parser policy, model inputs, or live
   resident state.
4. Preserve existing source-index policy identities. The helper lives outside
   the parser-policy component inventory; `NativeGraphSource` parsing and its
   cache construction remain unchanged.

## Natlang-native decomposition

The policy decision is absent: the same exact ordered rows are accepted or
rejected by the existing digest equality. The helper is crisp because exact
canonical byte serialization and SHA-256 are durability mechanisms. Existing
natural-language stages, graph validation and training behavior do not change.

## Verification

- Compare the helper against the established whole-value digest on empty,
  nested, Unicode, tuple, boolean, numeric, and floating-point sequences.
- Measure peak allocation on a large synthetic list and verify streaming uses
  less temporary memory while producing the identical digest.
- Exercise resident full-entry call sites and retain all existing mutation and
  append checks.
- Validate the already-prepared D31 QA request through cloned R19 and R18
  consumers using the shared helper, with no live adoption or exposure claim.
