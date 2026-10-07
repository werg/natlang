---
description: Front end, semantic analysis of a typed Python program into an LLVM module frame.
args:
  source: string
  problem?: string
returns: ModuleFrame
---
source is a Python program in a typed subset: functions annotated with int, float, bool, str, list[int] and
list[float]; module-level statements; integers that fit in 64 bits; no classes, exceptions or imports. Do what a
compiler front end's semantic analysis does: infer the type of every variable (each keeps one type), resolve names,
and report what falls outside the subset or would raise TypeError/NameError, with its line, in diagnostics.

Represent int as i64, float as double, bool as i1, str literals as constant C strings, and both list types as a
`ptr` to `%list = type { i64, i64, ptr }` (length, capacity, element data, 8 bytes per element). Build the header
as LLVM 22 IR text with opaque pointers: the `%list` type; globals for module-level variables that functions use;
every string literal and every printf format the program needs as `@.str.N` constants, with a comment quoting
each; `declare` lines for printf and scanf; and a `declare` for each runtime function the program needs, each with
a comment stating its contract, from this set: `@rt_list_new(i64 capacity) -> ptr`, `@rt_list_append(ptr, i64
bits)` (a float element is stored as its bit pattern), `@rt_list_get(ptr, i64 index) -> i64` and
`@rt_list_set(ptr, i64 index, i64 bits)` (negative indices count from the end; out of range prints `IndexError`
and exits with status 1), `@rt_floordiv(i64, i64) -> i64` and `@rt_mod(i64, i64) -> i64` (Python's floor
semantics), `@rt_read_int() -> i64` (the next integer on stdin, like `int(input())` or one token of `input().split()`).

List every function with its name, source text and `define` signature; the module-level statements become the
function `main`, with signature `define i32 @main()` and those statements as its source. You can check the header
with toolchain.verify. problem, when given, says why an earlier answer to this same request was rejected (the
verifier's message on its header); make sure your answer does not have it.
