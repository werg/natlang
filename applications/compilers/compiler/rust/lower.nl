---
description: Front end, IR generation for one Rust function from its checked syntax tree, unoptimized, with release-build semantics.
args:
  fn: SourceFunction
  context: string
  problem?: string
returns: string
---
Generate LLVM 22 IR for fn by walking fn.tree, the way an unoptimized compiler does. Start the definition with
fn.signature.

1. Entry block: an `alloca` for each parameter and each binding (a shadowing `let` is a new binding with its own slot),
   then a store of each parameter.
2. Expressions with values. A block's value is its `value` expression. `if`, `match` and `loop` with `break` values
   store each arm's value to a slot of their own and load it at the join. `match` on integers is a `switch`, and `_`
   is its default.
3. Loops. `while` and `loop` branch as usual. `for i in a..b` counts up while `i < b` (`<=` for `..=`). `.rev()` counts
   down from the end, and `.step_by(k)` adds k. `for x in &v` indexes the vector from 0 to its length, read once.
   `break` and `continue` branch to the loop's end and its next step.
4. Arithmetic as a release build has it. Integer `add`, `sub` and `mul` wrap: no `nsw` or `nuw`. `/` and `%` truncate
   toward zero: `sdiv` and `srem` for signed types, `udiv` and `urem` for unsigned ones. A zero divisor calls
   `@rt_panic` first. `as` truncates, extends (sign or zero, by the source type) or converts. `&&` and `||`
   short-circuit.
5. Places. A struct field is a `getelementptr` into the struct. A method call passes the receiver's address first.
   Vec operations call the runtime: `@rt_vec_new`, `push`, `pop`, `get`, `set`, `swap`, and `len` as a load of the
   length field. An f64 element goes through `bitcast`.
6. Output and input. `println!` and `print!` call printf with the header's formats; a bool prints as `true` or
   `false`. Reading a number is `@rt_read_i64`. `main` returns 0.

context holds the module's types, globals, format constants, and the declarations of the runtime and of every function,
each with its contract; use those names. Check your function with `toolchain.verify(context + "\n" + yourFunction)`
and fix what the verifier reports. Answer with the definition only, from `define` to its closing brace.

problem, when given, says why an earlier answer was rejected (the verifier's message, or how the program's output
changed); make sure your answer does not have it.
