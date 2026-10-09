/** Backward-compatible teacher API; reusable checks live outside teacher tooling. */
export * from '../evaluation/oracles.js';
// Registers the bundled benchmark comparators (tatqa-answer-record, ...) that specs may name.
import '../benchmarks/builtin.js';
