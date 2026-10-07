/** A translation unit read by a front end: what every function needs, and the functions themselves. */
export type ModuleFrame = {
  /**
   * LLVM IR for everything but function bodies: named struct types, globals with initializers, every string
   * literal as a private constant (`@.str.N`, with a comment quoting it), and `declare` lines for each library or
   * runtime function the program calls.
   */
  header: string,
  /** Each function the program defines, in source order. */
  functions: SourceFunction[],
  /** Problems that make the program invalid (unknown names, type errors), each with its line; empty when valid. */
  diagnostics: string[],
};

/** One function of the source program. */
export type SourceFunction = {
  name: string,
  /** Its LLVM `define` line up to the opening brace, e.g. `define i64 @fib(i64 %n)`. */
  signature: string,
  /** Its source text, exactly as written. */
  source: string,
};

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
