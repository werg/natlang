---
description: Front end, semantic analysis. Resolve, type-infer and borrow-check a Rust syntax tree, as rustc's front end does, and report errors.
args:
  syntax: Syntax
returns: Checked
---
First look at syntax's diagnostics, which the parser filled in. When it has any, return them unchanged as the
diagnostics, with no declarations, and analyze nothing. Otherwise analyze syntax as follows.

Analyze syntax, a parsed Rust program in the subset this compiler takes, and return its declarations with every node
analyzed, and its errors. The subset:
- free functions and `main`; structs of scalar fields, with `impl` blocks;
- the scalar types i32, i64, u32, u64, usize, f64 and bool, and `Vec<T>` of them (`Vec::new`, `with_capacity`,
  `vec![x; n]`, `push`, `pop`, `len`, indexing, `swap`, `iter`, `for x in &v`);
- `let` and `let mut` with shadowing; `if`, `match` on integers and `loop` with `break` values as expressions;
  `while`; `for` over ranges (`a..b`, `a..=b`, `.rev()`, `.step_by(k)`) and vectors;
- `as` casts; `wrapping_add`, `wrapping_sub` and `wrapping_mul`; `sqrt` and `abs` on f64;
- `println!` and `print!` with `{}` and `{:.N}`;
- reading stdin with `read_line` and `trim().parse::<T>().unwrap()`, also over `split_whitespace()`.
There are no traits, closures, other generics, strings or modules.

1. Name resolution. Resolve every path to its item, binding or method, respecting shadowing and block scope. ref is the
   binding's kind and line ("Local 5", "ItemFn 1", "ImplItemFn 12").
2. Type inference. Infer each binding's and expression's type as rustc does. An integer literal takes the type its use
   requires, else i32. Method calls resolve against the receiver's type, with auto-referencing. `if`, `match` and
   `loop` have the type of their arms, and every arm must agree.
3. Borrow checking. Track each binding's moves and borrows through the control flow. Report:
   - a use of a moved value (a struct or Vec moved by assignment or by a call taking it by value);
   - a mutable borrow while another borrow is live;
   - assignment to an immutable binding or through a shared reference;
   - a borrow that outlives its referent.
4. Errors, each as rustc would report it, with its line: also an unknown name, mismatched types, a missing method, and
   a construct outside the subset.

Analyze each function and method on its own, all at once, once the items are known.
