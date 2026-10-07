---
description: Front end, IR generation for the module. Lay out a checked Python program's types, globals, constants and runtime contracts as an LLVM module header, and give each function its LLVM signature.
args:
  checked: Checked
  problem?: string
returns: ModuleFrame
---
Generate the module-level LLVM 22 IR for checked, an analyzed Python program, with opaque pointers. int is i64, float
double, bool i1, str a constant C string, and both list types a `ptr` to `%list = type { i64, i64, ptr }` (length,
capacity, element data; 8 bytes per element, a float stored as its bit pattern).

The header holds:
- the `%list` type when the program uses lists;
- a global for each module-level variable that functions read (`@total = global i64 0`);
- every string literal and every printf format the prints need as `@.str.N` constants (`%ld`, `%s`, `True`,
  `False`, a space, a newline), each with a comment quoting it;
- `declare` lines for printf and scanf;
- a `declare` for each runtime function the program needs, each with a comment stating its contract, from this set:
  - `@rt_list_new(i64 capacity) -> ptr`;
  - `@rt_list_append(ptr, i64 bits)`, where a float element is stored as its bit pattern;
  - `@rt_list_get(ptr, i64 index) -> i64` and `@rt_list_set(ptr, i64 index, i64 bits)`: negative indices count from the
    end, and out of range they print `IndexError` and exit with status 1;
  - `@rt_floordiv(i64, i64) -> i64` and `@rt_mod(i64, i64) -> i64`, with Python's floor semantics;
  - `@rt_read_int() -> i64`: the next integer on stdin, like `int(input())` or one token of `input().split()`;
  - `@rt_print_float(double)`: print the value as Python's repr does, with no newline. That is the shortest of
    `%.15g`, `%.16g` and `%.17g` that reads back as the same double, with `.0` added when the text has no `.`, `e`,
    `inf` or `nan`.

functions: each FunctionDef in order, with its name, its `define` signature from its annotations, and its checked
tree. The module-level statements become the function `main`, with signature `define i32 @main()`. Its tree is a
FunctionDef named main holding those statements. Check the header with toolchain.verify.

problem, when given, says why an earlier answer was rejected (the verifier's message on the header); make sure your
answer does not have it.
