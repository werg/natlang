---
description: Front end, IR generation for one function of a Rust program, unoptimized.
args:
  fn: SourceFunction
  context: string
  problem?: string
returns: string
---
Translate the Rust function fn into one LLVM 22 IR function definition that starts with fn.signature, the way an
unoptimized compiler does: an `alloca` for each local in the entry block, loads and stores for every use, one block
per control-flow region. Keep Rust's semantics as a release build has them: integer arithmetic wraps (plain `add`,
`sub` and `mul`, without `nsw` or `nuw`); `/` and `%` truncate toward zero (`sdiv`/`srem` for signed types,
`udiv`/`urem` for unsigned ones), and a zero divisor panics through `@rt_panic`; `as` casts truncate, extend or
convert by the types involved; a range excludes its end unless written `..=`; `&&` and `||` short-circuit; `Vec`
operations go through the runtime functions; `println!` prints through printf with the header's formats (a bool as
`true` or `false`); `main` returns 0.

context holds the module's types, globals, format constants (`@.str.N`, commented), and the declarations of the
runtime and of every function, each with its contract; use those names. Check your function with
`toolchain.verify(context + "\n" + yourFunction)` before answering, and fix what the verifier reports.

Answer with the function definition only, from `define` to its closing brace.

problem, when given, says why an earlier answer to this same request was rejected (the verifier's message, or how the
program's output changed); make sure your answer does not have it.
