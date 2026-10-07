---
description: Front end, semantic analysis of a C translation unit into an LLVM module frame.
args:
  source: string
returns: ModuleFrame
---
Read source, a C99 program for a 64-bit target (AArch64 Linux: int is i32, long and pointers are 64 bits, char is
signed i8), and do what a C front end's semantic analysis does: resolve every name and type, and report errors.

Build the frame's header as LLVM 22 IR text, with opaque pointers (`ptr`):
- each struct as a named type (`%struct.point = type { double, double }`);
- each global variable with its initializer;
- every string literal in the program as `@.str.N = private unnamed_addr constant [len x i8] c"...\00"`, numbered in
  order of appearance, with a comment on the same line quoting the C literal and naming the function that uses it;
- a `declare` line for every library function called (`declare i32 @printf(ptr, ...)`, `declare ptr @malloc(i64)`,
  `declare double @sqrt(double)`, …).

List every function definition with its name, its exact source text and its `define` signature (C types mapped
as above; `int main(void)` is `define i32 @main()`). Put each error that would stop a C compiler in diagnostics,
with its line; leave diagnostics empty for a valid program. You can check the header with toolchain.verify.
