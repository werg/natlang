/**
 * The commit of natlang's llama.cpp fork (werg/llama.cpp-neuralese, NATLANG_FORK.md there) that `natlang runtime build`
 * compiles: decision scoring, guided generation, native state spans, RL capture and backend sampling on top of
 * upstream. A build is identified by this exact commit, not by its build number.
 */
export const LLAMA_FORK_SOURCE = Object.freeze({
  repository: 'werg/llama.cpp-neuralese',
  branch: 'neuralese',
  commit: 'b536c7379b2e330cad40132eec55d755bf4ee4a8',
});
