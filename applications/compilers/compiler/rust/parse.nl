---
description: Front end, parser. Read a Rust program into its syntax tree, as syn parses it.
args:
  source: string
returns: Syntax
---
Parse source, a Rust 2021 program, into its syntax tree, with node kinds as syn names them. Each top-level item is one
declaration: an ItemFn (its signature with typed inputs and output as text, and its block), an ItemStruct (its fields
with their types), or an ItemImpl (its self type as text and its methods, each an ImplItemFn whose first input is
`self`, `&self` or `&mut self`).
- statements: Local (`let`, with `mut` and the pattern's name as text, its type when written, its initializer),
  Expr statements, and items;
- expressions: ExprBlock, ExprIf, ExprMatch (arms with integer or wildcard patterns), ExprWhile, ExprLoop, ExprForLoop
  (pattern, the iterated expression, body), ExprBreak (with a value), ExprContinue, ExprReturn, ExprAssign, ExprBinary
  and ExprUnary (operator as text, compound assignment included), ExprCall, ExprMethodCall (method name as text),
  ExprField, ExprIndex, ExprRange (`..` or `..=` as text), ExprCast (the type as text), ExprReference (`&` or `&mut`),
  ExprPath (the path as text), ExprLit, ExprStruct, and ExprMacro for `println!`, `print!` and `vec!` (format string and
  arguments as children).
A block's last expression without a semicolon is its value; mark that node's text `value`. type and ref stay null:
semantic analysis fills them.

Put each syntax error in diagnostics with its line; leave diagnostics empty for a program that parses.
