---
description: Front end, parser. Read a typed Python program into its syntax tree, as Python's ast module does.
args:
  source: string
returns: Syntax
---
Parse source, a Python 3 program, into its syntax tree, with node kinds and fields as `ast.parse` produces them.
Each top-level statement is one declaration: a FunctionDef (its arguments with their annotations as text, its return
annotation, its body), or a module-level statement (Assign, AnnAssign, AugAssign, Expr, For, While, If).
- statements: Assign, AnnAssign, AugAssign (operator as text), Return, If, For (target, iter, body), While, Break,
  Continue, Pass, Expr;
- expressions: BinOp, UnaryOp, BoolOp and Compare (operators as text; a chained comparison keeps all its parts),
  Call, Attribute (`xs.append`), Subscript, Name, Constant (the literal as written), List, and IfExp.
Indentation decides blocks. type and ref stay null: semantic analysis fills them.

Put each syntax error in diagnostics with its line (bad indentation, an unclosed bracket); leave diagnostics empty for
a program that parses.
