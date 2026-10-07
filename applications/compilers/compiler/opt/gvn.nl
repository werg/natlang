---
description: Middle end, global value numbering. Replace computations and loads that repeat a dominating one.
args:
  fn: string
  context: string
  flow: Flow
  problem?: string
returns: string
---
Eliminate redundancy in fn as LLVM's GVN does. flow is fn's control flow.

1. Walk the dominator tree from the entry in preorder, with a scoped table from expression to value number. Leaving a
   block's subtree drops what it added.
2. Each pure instruction (arithmetic, comparisons, casts, getelementptr, select) has an expression: its opcode, its
   flags, its type and its operands' value numbers, with the operands of commutative operations sorted. When the table
   already holds the expression from a dominating block, replace the instruction by that value and delete it.
   Otherwise add it.
3. Loads. A load of an address is redundant when one of these dominates it with nothing between them that may write
   that memory (a store to an address that may alias, or a call that is not known to be read-only):
   - an earlier load of the same address and type, whose value it takes;
   - a store to that address, whose stored value it takes.
   Two addresses may alias unless they are distinct allocas, distinct globals, or offsets from one base that differ by a
   constant.
4. A phi whose incoming values all have the same number is that value.

context holds the module's types, globals and function declarations. Check the result with
`toolchain.verify(context + "\n" + result)` and fix what the verifier reports. Answer with the transformed function
only (same signature and behavior); if nothing applies, answer fn unchanged.

problem, when given, says why an earlier answer was rejected (the verifier's message, or how the program's output
changed); make sure your answer does not have it.
