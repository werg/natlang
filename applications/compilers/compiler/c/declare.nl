---
description: Front end, IR generation for the module. Lay out a checked C program's types, globals, strings and declarations as an LLVM module header, and give each function its LLVM signature.
args:
  checked: Checked
  problem?: string
returns: ModuleFrame
---
Generate the module-level LLVM 22 IR for checked, an analyzed C program for AArch64 Linux, with opaque pointers
(`ptr`), as clang's CodeGen does before it emits function bodies. Map types as clang does: int i32, long i64, char
i8, double double, any pointer ptr, a struct its named type.

The header holds:
- each struct as a named type, fields in order (`%struct.point = type { double, double }`);
- each global variable with its initializer, constant-folded (`@count = global i32 0`);
- every string literal in the program as `@.str.N = private unnamed_addr constant [len x i8] c"...\00"`, numbered in
  order of appearance, with a comment on the same line that quotes the C literal and names the function using it;
- a `declare` line for every library function called (`declare i32 @printf(ptr, ...)`, `declare ptr @malloc(i64)`,
  `declare double @sqrt(double)`).

functions: for each FunctionDecl with a body, in order, its name, its `define` signature (`int main(void)` is `define
i32 @main()`, a parameter `%name` per ParmVarDecl), and its checked tree as given. diagnostics: empty unless the
module cannot be laid out. Check the header with toolchain.verify.

problem, when given, says why an earlier answer was rejected (the verifier's message on the header); make sure your
answer does not have it.
