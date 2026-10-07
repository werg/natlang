---
description: Front end, semantic analysis of a Rust program (a subset) into an LLVM module frame, borrow checking included.
args:
  source: string
  problem?: string
returns: ModuleFrame
---
source is a Rust 2021 program in a subset: free functions and `main`; structs of scalar fields with `impl` blocks
(methods taking `&self`, `&mut self` or `self`); the scalar types i32, i64, u32, u64, usize, f64 and bool; `Vec<T>`
of those scalars (`Vec::new`, `Vec::with_capacity`, `vec![x; n]`, `push`, `pop`, `len`, indexing, `swap`, `for x in
&v` and `.iter()`); `let` and `let mut` with shadowing; `if`/`else` and `match` on integers as expressions; `while`,
`loop` with `break` values, `for i in a..b`, `a..=b`, `(a..b).rev()` and `.step_by(k)`; `as` casts; `wrapping_add`,
`wrapping_sub` and `wrapping_mul` on integers; `sqrt` and `abs` on f64; `println!` and
`print!` with `{}` for integers and bools and `{:.N}` for f64; reading stdin with `std::io::stdin().read_line(&mut
line)` and `line.trim().parse::<T>().unwrap()` (also over `split_whitespace()`). No traits, generics beyond `Vec<T>`,
closures, other strings or modules.

Do what rustc's front end does: resolve every name, infer every type, and check the program as the borrow checker
would. Put each error rustc would report (an unknown name, mismatched types, a use of a moved value, a mutable
borrow while another borrow is live, assignment to an immutable binding) in diagnostics, with its line; leave
diagnostics empty for a valid program.

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

List every function, each method as its own function named `Type_method` (`@Point_dist`), with its name, its source
text and its `define` signature; `fn main()` is `define i32 @main()`. You can check the header with toolchain.verify.
problem, when given, says why an earlier answer to this same request was rejected (the verifier's message on its
header); make sure your answer does not have it.
