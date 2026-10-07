---
description: Front end, parser. Read a C translation unit into its syntax tree, as clang's parser does.
args:
  source: string
returns: Syntax
---
Parse source, a C99 program, into its syntax tree. Each top-level declaration is one node: a FunctionDecl, a
VarDecl (a global), a RecordDecl (a struct) or a TypedefDecl. `#include` lines name headers and leave no node. Build
the nodes as clang's parser does:
- declarations: FunctionDecl with its ParmVarDecls and its body; VarDecl with its type as written in text and its
  initializer.
- statements: CompoundStmt, DeclStmt, IfStmt, ForStmt (init, condition, increment, body; an absent part is a
  NullStmt), WhileStmt, DoStmt, ReturnStmt, BreakStmt, ContinueStmt, SwitchStmt with CaseStmt/DefaultStmt.
- expressions, with C's precedence and associativity: BinaryOperator and UnaryOperator (the operator as text),
  CompoundAssignOperator, ConditionalOperator, CallExpr, ArraySubscriptExpr, MemberExpr (text `.x` or `->x`),
  CStyleCastExpr (the type as text), DeclRefExpr (the name as text), and IntegerLiteral, FloatingLiteral,
  CharacterLiteral, StringLiteral (the literal as written).
type and ref stay null: semantic analysis fills them.

Parse each function's body on its own, all of them at once, after listing the top-level declarations. Put each
syntax error in diagnostics with its line (a missing semicolon, an unbalanced brace); leave diagnostics empty for a
program that parses.
