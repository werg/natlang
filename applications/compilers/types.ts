/**
 * A syntax tree node, as the language's reference parser builds it. kind is the grammar's name for the construct (for C
 * as clang names them: FunctionDecl, ParmVarDecl, CompoundStmt, IfStmt, BinaryOperator; for Python as its ast module
 * does: FunctionDef, For, BinOp; for Rust as syn does: ItemFn, ExprIf, ExprMethodCall). line is its source line;
 * text is the token it stands for (an operator, a name, a literal, a type as written), or null; children are its parts
 * in source order. Semantic analysis fills type (the node's type in the language's own terms, such as `int`,
 * `list[float]` or `&mut Vec<i64>`) and ref (for a name, the kind and line of the declaration it refers to, such as
 * "ParmVarDecl 3"), and adds the conversions the language makes implicitly as nodes of their own (ImplicitCastExpr
 * IntegralCast, LValueToRValue).
 */
export type Node = { kind: string, line: number, text: string | null, type: string | null, ref: string | null, children: Node[] };

/** A parsed program: its top-level declarations in source order, and its syntax errors, each with its line. */
export type Syntax = { declarations: Node[], diagnostics: string[] };

/**
 * A program after semantic analysis: the same declarations, every expression typed and every name resolved, and each
 * error that makes the program invalid (an unknown name, a type error, a borrow error), with its line; empty when valid.
 */
export type Checked = { declarations: Node[], diagnostics: string[] };

/** A translation unit ready for IR generation: what every function needs, and the functions themselves. */
export type ModuleFrame = {
  /**
   * LLVM IR for everything but function bodies: named struct types, globals with initializers, every string
   * literal as a private constant (`@.str.N`, with a comment quoting it), and `declare` lines for each library or
   * runtime function the program calls.
   */
  header: string,
  /** Each function the program defines, in source order. */
  functions: SourceFunction[],
  /** Problems found while laying out the module; empty when there are none. */
  diagnostics: string[],
};

/** One function of the source program, as IR generation needs it. */
export type SourceFunction = {
  name: string,
  /** Its LLVM `define` line up to the opening brace, e.g. `define i64 @fib(i64 %n)`. */
  signature: string,
  /** Its checked syntax tree (a FunctionDecl, FunctionDef or ItemFn; a method as its own function). */
  tree: Node,
};

/**
 * Control-flow facts about one IR function, as LLVM's analyses compute them. blocks: each basic block by its label
 * (the entry block first, then in order), with its successors and predecessors, its immediate dominator (null for
 * the entry), and its dominance frontier. loops: the natural loops, innermost first. Each has its header, all of
 * its blocks, its latches (the blocks that branch back to the header), its exits (the blocks outside it that it
 * branches to), its preheader (the header's only predecessor outside the loop, or null when there is none) and its
 * depth (1 for an outermost loop).
 */
export type Flow = {
  blocks: { name: string, successors: string[], predecessors: string[], idom: string | null, frontier: string[] }[],
  loops: { header: string, blocks: string[], latches: string[], exits: string[], preheader: string | null, depth: number }[],
};

/**
 * Machine code for one function between instruction selection and emission: one AArch64 instruction per line, in GNU
 * syntax, under the function's block labels (`.L<function>_<n>:`). Before register allocation, values live in virtual
 * registers: %x<n> for 64-bit and %w<n> for 32-bit integers (the same n is the same register), and %d<n> for doubles.
 * Physical registers appear only where the calling convention fixes them (arguments, results, calls).
 * Stack objects, an alloca or a spill slot, are frame indices `[fi#k]`, each declared on a line
 * `; fi#k: <size> bytes, align <a>` at the top, until frame lowering gives them offsets.
 */
export type MachineCode = string;

/**
 * Live intervals of a machine function's virtual registers. Instructions are numbered from 0 in order, labels
 * excluded. A register is live from its first definition to its last use, extended over a whole loop when it is
 * live around the loop's back edge. crossesCall is true when a call (`bl`, `blr`) lies strictly inside the interval.
 * calls are the numbers of the call instructions.
 */
export type Liveness = { intervals: { register: string, start: number, end: number, crossesCall: boolean }[], calls: number[] };

/** Middle-end passes, as the pass manager names them. */
export type Pass = 'mem2reg' | 'simplify' | 'gvn' | 'licm' | 'loops' | 'inline' | 'dce';
export type Level = 'O1' | 'O2' | 'O3';

/** What the natural-language pass manager (`compiler.nl`) produces. */
export type Compiled = {
  /** The optimized module: the header and every function, as LLVM IR text. */
  ir: string,
  /** The AArch64 program: the data section and every function; empty when the back end did not finish. */
  assembly: string,
  /** Why the program could not be compiled (the front end's diagnostics, or a stage that never worked); empty on success. */
  diagnostics: string[],
  /** One line per stage and function: the stage, the function, accepted or kept the previous version, and why. */
  log: string[],
};
