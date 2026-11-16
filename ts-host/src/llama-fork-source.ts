/**
 * The commit of natlang's llama.cpp fork (werg/llama.cpp-neuralese, NATLANG_FORK.md there) that `natlang runtime build`
 * compiles: decision scoring, guided generation, native state spans, RL capture and backend sampling on top of
 * upstream. A build is identified by this exact commit, not by its build number.
 */
export const LLAMA_FORK_SOURCE = Object.freeze({
  repository: 'werg/llama.cpp-neuralese',
  branch: 'neuralese',
  commit: 'b704a8b2f2b28c41576dfc9fd0a98f6f674194d7',
});
