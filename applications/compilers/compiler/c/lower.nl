---
description: Front end, IR generation for one C function from its checked syntax tree, unoptimized, as clang -O0 emits it.
args:
  fn: SourceFunction
  context: string
  problem?: string
returns: string
---
Generate LLVM 22 IR for fn by walking fn.tree, as clang's CodeGen does at -O0. Start the definition with fn.signature.

1. Entry block: an `alloca` for each parameter and each local variable (each VarDecl in the body, even in inner
   blocks), then a store of each parameter into its slot.
2. Statements, in order. A CompoundStmt emits its statements. An IfStmt branches on its condition to a then-block and
   an else-block (or straight to the join), which then branch to a join block. A ForStmt and a WhileStmt get blocks for
   the condition, the body, the increment and the end; `continue` branches to the increment (to the condition in a
   while loop) and `break` to the end. A DoStmt tests at the bottom. A SwitchStmt is a `switch` on its value, with each
   case's block falling through to the next. A ReturnStmt stores to a return slot and branches to the single return
   block.
3. Expressions, from the tree's types and implicit-conversion nodes. An LValueToRValue is a `load`; integer
   conversions are `sext`, `zext` or `trunc`; IntegralToFloating and back are `sitofp` and `fptosi`. Signed arithmetic
   carries `nsw`; `/` and `%` are `sdiv` and `srem`. Array indexing and members are `getelementptr` with the struct's
   layout. Comparisons are `icmp` or `fcmp` followed by a `zext` to int. `&&` and `||` short-circuit through blocks and a
   `phi`. A call passes its converted arguments; a string literal is its `@.str.N` from context.
4. Every block ends in exactly one terminator; a block left without one after a return or break gets none of its own
   code and is removed.

context holds the module's types, globals, strings and the declarations of every function; use those names. Check
your function with `toolchain.verify(context + "\n" + yourFunction)` and fix what the verifier reports. Answer with the
definition only, from `define` to its closing brace.

problem, when given, says why an earlier answer was rejected (the verifier's message, or how the program's output
changed); make sure your answer does not have it.
