---
description: Front end, IR generation for one Python function from its checked syntax tree, unoptimized.
args:
  fn: SourceFunction
  context: string
  problem?: string
returns: string
---
Generate LLVM 22 IR for fn by walking fn.tree, the way an unoptimized compiler does. Start the definition with
fn.signature.

1. Entry block: an `alloca` for each parameter and each local, then a store of each parameter into its slot.
2. Statements. If and While branch through blocks as in C. `for x in range(a, b, s)` counts from a while `x < b` for
   a positive step and `x > b` for a negative one. `for x in xs` indexes the list from 0 to its length, read once.
   Break and continue branch to the loop's end and its next step.
3. Expressions, by their types from analysis. int arithmetic is plain i64 `add`, `sub` and `mul`. `//` and `%` call
   `@rt_floordiv` and `@rt_mod`. `/` converts both sides to double. `and` and `or` short-circuit through blocks and a
   `phi`. `len(xs)` loads the list's length. Indexing calls `@rt_list_get` (a float comes back as bits, made a
   double with `bitcast`). `xs.append(v)` and an index assignment call `@rt_list_append` and `@rt_list_set`. A list
   display makes a list and appends each element. `int(input())` is `@rt_read_int`.
4. `print(a, b, …)` prints each value with its format, separated by single spaces and ended by a newline:
   - an int with `%ld`;
   - a float through `@rt_print_float`;
   - a str with `%s`;
   - a bool as `True` or `False`.
5. A function without a return statement returns 0 of its type at its end; `main` returns 0.

context holds the module's types, globals, constants, and the declarations of the runtime and of every function, each
with its contract; use those names. Check your function with `toolchain.verify(context + "\n" + yourFunction)` and fix
what the verifier reports. Answer with the definition only, from `define` to its closing brace.

problem, when given, says why an earlier answer was rejected (the verifier's message, or how the program's output
changed); make sure your answer does not have it.
