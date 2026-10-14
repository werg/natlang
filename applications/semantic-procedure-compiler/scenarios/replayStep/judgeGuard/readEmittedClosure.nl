---
description: Discover actual generated code and runtime-read closure without ideal procedure guidance.
args:
  entry: GeneratedModule
  program: GeneratedProgram
  selectedClauses: SourceClause[]
returns: EmittedClosure
---
1. Project program.modules and program.typeFiles into immutable ExecutionFiles containing only path,
   text and module/types kind. Do not project plans, source-to-module promises or contribution audits.
   Require distinct paths and entry.path/text to equal its actual supplied original module.
2. Initialize CodeClosureState with pendingPaths containing entry.path, unvisitedPaths containing
   every original file path, empty visited files/references/runtime reads and original selectedClauses.
   These clauses are available runtime data, not behavior guidance; none enter the closure initially.
3. Call followCodeReference.iterateOn with measure unvisitedPaths.length. Its finished predicate is
   no pending path remains or an explicit unresolved dependency prevents a complete closure.
   A child may activate only an unvisited original file, so discovery never enlarges the file universe.
4. Consume the actual child files, references and runtimeReads. Require every resolved code reference
   to connect visited original files, and every runtime read to have its exact supporting generated span.
   A cycle is a revisited existing node, not a fabricated extra file or a reason to erase an edge.
5. Build runtimeClauses from the union of actually returned clause IDs. A returned wholeCollection
   read authorizes the actual selectedClauses collection; no unreferenced source policy is supplied.
   Validate IDs against the original selected collection and preserve complete original source values.
6. Return entryPath, files, references, runtimeReads, runtimeClauses and unresolved. Preserve actual
   generated caller/callee attribution. judgeGuard uses this returned closure for emitted interpretation;
   required-source judgment receives its own independently prepared context.
