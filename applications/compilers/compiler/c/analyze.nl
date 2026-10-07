---
description: Front end, semantic analysis. Resolve names and type a C syntax tree, as clang's Sema does, and report errors.
args:
  syntax: Syntax
returns: Checked
---
Analyze syntax, a parsed C99 translation unit for AArch64 Linux, where int is 32 bits, long and pointers 64, and
char is signed. Return its declarations with every node analyzed, and its errors.

1. Scopes. Walk the declarations in order, with a scope for the file, each function and each block. Give every
   DeclRefExpr the declaration it names as ref ("ParmVarDecl 3"); a library function the program calls without
   declaring it (printf, malloc, sqrt) refers to its standard declaration ("stdio.h printf").
2. Types. Give every expression its C type: literals by their form, names by their declarations, operators by C's
   rules (usual arithmetic conversions, pointer arithmetic, comparisons are int), calls by the callee's return type,
   members by the struct's field.
3. Implicit conversions. Make each one a node of its own, as clang does: an ImplicitCastExpr with text
   LValueToRValue where a variable's value is read, IntegralCast for integer promotions and conversions, IntegralToFloating and
   FloatingToIntegral, ArrayToPointerDecay, FunctionToPointerDecay. Default argument promotions to a variadic call
   make float double and char or short int.
4. Errors. Report each problem a C compiler stops at, with its line: an undeclared name, a call with the wrong number of
   arguments, assigning to something that is not an lvalue, a type that does not fit (a struct where a number is
   needed), `break` outside a loop or switch.

Analyze each function body on its own, all at once, once the file scope is known.
