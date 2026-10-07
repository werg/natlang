/** Validate output-value boundary hints against the task's own scoped source. */
export function validateSourceValueBoundaries(record) {
  const id = record?.id ?? 'source row';
  const folderFiles = record?.semantics?.folder_files;
  const actualTaskText = folderFiles?.['task.json'];
  const expectedTaskText = record?.semantics?.expected_files?.['task.json'];
  if (typeof actualTaskText !== 'string') {
    if (typeof expectedTaskText === 'string')
      throw new Error(`${id}: expected task.json exists but actual folder FileHandle task.json is missing`);
    return { status: 'no_boundary_contract', fields: [], boundary_fields: [] };
  }
  let task;
  try { task = JSON.parse(actualTaskText); }
  catch (error) { throw new Error(`${id}: actual folder task.json is invalid JSON: ${error.message}`); }
  if (typeof expectedTaskText === 'string') {
    let expectedTask;
    try { expectedTask = JSON.parse(expectedTaskText); }
    catch (error) { throw new Error(`${id}: expected task.json is invalid JSON: ${error.message}`); }
    if (canonical(task) !== canonical(expectedTask))
      throw new Error(`${id}: actual folder task.json differs from expected_files task.json`);
  }

  const contract = task.output_contract;
  const boundaryMap = contract?.field_value_boundaries;
  if (boundaryMap === undefined || boundaryMap === null ||
      (typeof boundaryMap === 'object' && !Array.isArray(boundaryMap) && Object.keys(boundaryMap).length === 0))
    return { status: 'no_boundary_contract', fields: [], boundary_fields: [] };
  const fields = contract?.fields;
  if (!fields || typeof fields !== 'object' || Array.isArray(fields))
    throw new Error(`${id}: output_contract.fields must be an object when field_value_boundaries are declared`);
  if (!boundaryMap || typeof boundaryMap !== 'object' || Array.isArray(boundaryMap))
    throw new Error(`${id}: output_contract.field_value_boundaries must be an object`);

  const passes = task.passes;
  if (!Array.isArray(passes)) throw new Error(`${id}: task.passes must be an array when field_value_boundaries are declared`);
  if (!folderFiles || typeof folderFiles !== 'object') throw new Error(`${id}: missing source FileHandle contents`);

  for (const [field, rule] of Object.entries(boundaryMap)) {
    if (!Object.hasOwn(fields, field))
      throw new Error(`${record.id}: boundary field ${JSON.stringify(field)} is not declared in output_contract.fields`);
    const spans = rule?.pass_value_spans;
    if (!Array.isArray(spans) || spans.length === 0)
      throw new Error(`${record.id}: boundary ${JSON.stringify(field)} must declare nonempty pass_value_spans`);
    for (const [spanIndex, span] of spans.entries()) {
      const label = `${id}: boundary ${JSON.stringify(field)} span ${spanIndex}`;
      const matches = passes.filter(pass => pass?.name === span?.pass_name);
      if (matches.length !== 1)
        throw new Error(`${label} refers to pass ${JSON.stringify(span?.pass_name)}; found ${matches.length} exact pass matches`);
      const pass = matches[0];
      if (pass.evidence_path !== span.evidence_path)
        throw new Error(`${label} evidence_path ${JSON.stringify(span.evidence_path)} does not match pass evidence_path ${JSON.stringify(pass.evidence_path)}`);
      if (!Array.isArray(pass.allowed_fields) || !pass.allowed_fields.includes(field))
        throw new Error(`${label} field is not allowed by pass ${JSON.stringify(pass.name)}`);
      const evidence = folderFiles[pass.evidence_path];
      if (typeof evidence !== 'string')
        throw new Error(`${label} current-pass evidence ${JSON.stringify(pass.evidence_path)} is absent from this source row`);
      if (typeof span.start_after !== 'string' || span.start_after.length === 0 ||
          typeof span.end_before !== 'string' || span.end_before.length === 0)
        throw new Error(`${label} must use nonempty start_after and end_before anchors`);
      const startAt = evidence.indexOf(span.start_after);
      if (startAt < 0 || evidence.indexOf(span.start_after, startAt + 1) >= 0)
        throw new Error(`${label} start_after anchor must occur exactly once in its current-pass evidence`);
      const valueStart = startAt + span.start_after.length;
      const endAt = evidence.indexOf(span.end_before, valueStart);
      if (endAt < 0 || evidence.indexOf(span.end_before, endAt + 1) >= 0)
        throw new Error(`${label} end_before anchor must occur exactly once after start_after in its current-pass evidence`);
      if (endAt <= valueStart)
        throw new Error(`${label} anchors do not enclose a nonempty source value`);
    }
  }
  return { status: 'validated', fields: Object.keys(fields), boundary_fields: Object.keys(boundaryMap) };
}

function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object')
    return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${canonical(value[key])}`).join(',')}}`;
  return JSON.stringify(value);
}
