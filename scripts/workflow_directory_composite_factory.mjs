/**
 * Safe component selection and exact file embedding for workflow directory composites.
 * A component must be an individual question over an explicit state. Composite source
 * directory tasks must be expanded by a source-aware builder instead of nested here.
 */
export function standaloneComponentProblem(ir) {
  const sem = ir?.semantics;
  if (!sem || !sem.root || !sem.files?.[sem.root]) return 'missing_root_question';
  if (!sem.inputs || !Object.hasOwn(sem.inputs, 'state')) return 'missing_explicit_state';
  if (Object.hasOwn(sem, 'folder_files') || Object.hasOwn(sem, 'expected_files')) return 'nested_directory_component';
  if (/^kind:\s*directory-reducer\s*$/m.test(sem.files[sem.root])) return 'directory_reducer_component';
  if (!Object.hasOwn(sem, 'expected')) return 'missing_component_gold';
  return null;
}

export function componentFileRecord(ir) {
  const problem = standaloneComponentProblem(ir);
  if (problem) throw new Error(`workflow component ${ir?.id ?? '<unknown>'} is not standalone: ${problem}`);
  return {
    source_question_id: ir.source_ids?.[0] ?? ir.id,
    question: ir.semantics.files[ir.semantics.root],
    state: ir.semantics.inputs.state,
  };
}

export function buildQuestionBatchPayload(items, keys) {
  if (!Array.isArray(items) || !items.length || items.length !== keys.length) {
    throw new Error('items and keys must be nonempty arrays of equal length');
  }
  const files = {};
  const expected = {};
  for (let i = 0; i < items.length; i += 1) {
    const record = componentFileRecord(items[i]);
    const key = keys[i];
    if (typeof key !== 'string' || !key || Object.hasOwn(expected, key)) throw new Error(`invalid or duplicate key: ${key}`);
    files[`jobs/${key}.json`] = `${JSON.stringify(record)}\n`;
    expected[key] = items[i].semantics.expected;
  }
  return { folder_files: files, expected_files: structuredClone(files), expected };
}
