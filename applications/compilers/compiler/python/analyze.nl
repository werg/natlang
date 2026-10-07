---
description: Front end, semantic analysis. Type a Python syntax tree in the compiled subset and report what falls outside it.
args:
  syntax: Syntax
returns: Checked
---
Analyze syntax, a parsed Python program in the subset this compiler takes:
- functions annotated with int, float, bool, str, list[int] and list[float];
- module-level statements;
- integers that fit in 64 bits;
- no classes, exceptions, imports, closures or recursion through globals that change type.
Return its declarations with every node analyzed, and its errors.

1. Names. Resolve each Name to its binding as Python's scoping does: a parameter, a local (assigned in the function), a
   module-level variable, a function, or a builtin (print, len, range, int, float, input, abs, min, max). ref is
   the binding's kind and line ("arg 4", "local 6", "global 1", "builtin print").
2. Types. Annotations fix parameters and results. A local takes the type of its first assignment, and every later
   assignment must keep it. Infer each expression's type: int and float arithmetic (int with float is float; `/` is
   float; `//` and `%` keep int), comparisons and `not` are bool, indexing a list gives its element type,
   `input().split()` gives a list of str to convert, `range` gives int.
3. Errors, with their lines: a name used before any assignment, a variable that changes type, an operation Python
   would reject with TypeError (adding str to int, indexing an int), a construct outside the subset (a class, try,
   import, a lambda, a list of mixed types), and a call with the wrong number of arguments.

Analyze each function on its own, all at once, once the module's names are known.
