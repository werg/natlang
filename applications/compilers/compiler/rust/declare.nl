---
description: Front end, IR generation for the module. Lay out a checked Rust program's types, formats and runtime contracts as an LLVM module header, and give each function and method its LLVM signature.
args:
  checked: Checked
  problem?: string
returns: ModuleFrame
---
Generate the module-level LLVM 22 IR for checked, an analyzed Rust program, with opaque pointers, as rustc's code
generation lays out a crate before it emits bodies.

Represent the program as rustc's release build does on AArch64 Linux: integers by their width (usize as i64, bool
as i1), f64 as double; each struct as a named type (`%Point = type { double, double }`), passed by pointer to a
method taking `&self` or `&mut self`; a `Vec<T>` as a `ptr` to `%vec = type { i64, i64, ptr }` (length, capacity,
data; 8 bytes per element, an f64 stored as its bit pattern, a bool as 0 or 1). Integer overflow wraps. Indexing out
of bounds, `unwrap` of a failed parse and division by zero panic.

Build the header as LLVM 22 IR text with opaque pointers: the struct types and `%vec`; every format the prints need
as a printf format constant `@.str.N` (`%ld`, `%lu`, `%.3f`, …), with a comment quoting the Rust format it stands
for; `declare` lines for printf; and a `declare` for each runtime function the program needs, with a comment stating
its contract, from this set: `@rt_vec_new(i64 capacity) -> ptr`, `@rt_vec_push(ptr, i64 bits)`, `@rt_vec_pop(ptr)
-> i64` (panics on an empty vector), `@rt_vec_get(ptr, i64 index) -> i64` and `@rt_vec_set(ptr, i64 index, i64
bits)` (out of range they panic with "index out of bounds: the len is L but the index is I"), `@rt_vec_swap(ptr, i64,
i64)`, `@rt_read_i64() -> i64` (the next whitespace-separated integer on stdin), and `@rt_panic(ptr message)` (prints
`thread 'main' panicked: ` and the message to stderr, then exits with status 101).

functions: every function in order, each method as its own function named `Type_method` (`@Point_dist`, its self
parameter first as a pointer to the struct), with its name, its `define` signature and its checked tree. `fn main()`
is `define i32 @main()`. diagnostics: empty unless the module cannot be laid out. Check the header with
toolchain.verify.

problem, when given, says why an earlier answer was rejected (the verifier's message on the header); make sure your
answer does not have it.
