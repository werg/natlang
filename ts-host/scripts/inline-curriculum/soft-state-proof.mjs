/** Validate a real Neuralese writer->reader relation from exact runtime graph events. */
export function validateExpectedReadCount({ actualCount, expectedPaths, childCallId }) {
  if (!Number.isInteger(actualCount) || !Array.isArray(expectedPaths))
    throw new Error('source-read counts need an integer actualCount and an expected path list');
  if (actualCount !== expectedPaths.length)
    throw new Error(`child ${childCallId} has ${actualCount} source reads, expected ${expectedPaths.length}`);
  return { expected_count: expectedPaths.length, actual_count: actualCount,
    no_source_read_expected: expectedPaths.length === 0 };
}

/** Resolve the writer's declared port against the immediate consumer's authored input declaration. */
export function resolveSoftStateArgument({ producerNextArgument, consumerExpectedArgument, fallback = 'prior' }) {
  if (consumerExpectedArgument && producerNextArgument && consumerExpectedArgument !== producerNextArgument)
    throw new Error(`producer declares next argument ${producerNextArgument}, but consumer declares ${consumerExpectedArgument}`);
  return consumerExpectedArgument ?? producerNextArgument ?? fallback;
}

/** Count exact source reads observed in child actions. FileHandle openings are not currently traced. */
export function summarizeSourceEvidence(proofCases) {
  if (!Array.isArray(proofCases)) throw new Error('source evidence summary requires a proof-case list');
  return proofCases.reduce((summary, proofCase) => {
    for (const child of proofCase.clean_child_reads ?? []) {
      summary.successful_source_reads += Array.isArray(child.source_reads) ? child.source_reads.length : 0;
    }
    return summary;
  }, { successful_source_reads: 0 });
}

/** Match a complete parameter declaration, rejecting unions and substring lookalikes. */
export function signatureHasExactParameter(signatureValue, argument, type) {
  const signature = String(signatureValue ?? '');
  const escapeRegex = value => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const exactParameter = new RegExp(`(?:^|\\(|, )${escapeRegex(argument)}: ${escapeRegex(type)}(?=, |\\))`);
  return exactParameter.test(signature);
}

/** Match a typed member on one inline record parameter, such as input.notes. */
export function signatureHasExactArgumentPath(signatureValue, argumentPath, type) {
  const parts = String(argumentPath ?? '').split('.');
  if (parts.length === 1) return signatureHasExactParameter(signatureValue, parts[0], type);
  if (parts.length !== 2 || parts.some(part => !/^[A-Za-z_$][\w$]*$/.test(part))) return false;
  const [root, member] = parts;
  const signature = String(signatureValue ?? '');
  const escapeRegex = value => value.replace(/[.*+?^${}()|[\\]\\]/g, '\\$&');
  const pattern = new RegExp(`(?:^|\\(|, )${escapeRegex(root)}: \\{\\s*[^{}]*\\b${escapeRegex(member)}: ${escapeRegex(type)}(?=\\s*[,}])`);
  return pattern.test(signature);
}

export function validateSoftStateEdge({ graph, actualValue, expectedType = 'Neuralese<string>', writerCallId,
  consumerCallId, consumerArgument, writerNode, expectedBodySha256, argumentMode = 'capture' }) {
  if (!['capture', 'typed_argument'].includes(argumentMode))
    throw new Error(`unsupported Neuralese argument mode ${argumentMode}`);
  if (!actualValue || typeof actualValue !== 'object' || actualValue.$neuralese?.type !== expectedType ||
      typeof actualValue.$neuralese?.id !== 'string')
    throw new Error(`writer ${writerCallId} did not return ${expectedType}`);
  const block = actualValue.$neuralese.id;
  const writer = graph.find(event => event.kind === 'block_write' && event.call_id === writerCallId && event.block === block &&
    (writerNode === undefined || event.node === writerNode));
  if (!writer || typeof writer.node !== 'string') throw new Error(`no actual block_write for ${block} from ${writerCallId}`);
  const invocation = graph.find(event => event.kind === 'invocation' && event.call_id === consumerCallId && event.phase === 'start');
  const invocationEdge = invocation?.inputs?.find(input => input.node === writer.node && input.block === block &&
    input.port === `arg:${consumerArgument}`);
  if (!invocationEdge) throw new Error(`no exact ${consumerArgument} invocation edge from ${writer.node} to ${consumerCallId}`);
  const argumentParts = String(consumerArgument).split('.');
  const capturePort = argumentMode === 'capture' && argumentParts.length === 2 ? `capture:${argumentParts[1]}` : undefined;
  const captureEdge = capturePort && invocation?.inputs?.find(input => input.node === writer.node && input.block === block &&
    input.port === capturePort);
  if (capturePort && !captureEdge) throw new Error(`no exact ${capturePort} capture edge from ${writer.node} to ${consumerCallId}`);
  const signature = String(invocation.signature ?? '');
  if (!signatureHasExactArgumentPath(signature, consumerArgument, expectedType))
    throw new Error(`consumer ${consumerCallId} does not declare ${consumerArgument}: ${expectedType} (${signature})`);
  if (expectedBodySha256 !== undefined && writer.text_body_sha256 !== expectedBodySha256)
    throw new Error(`writer ${writerCallId} body digest does not match ${expectedBodySha256}`);
  const read = graph.find(event => event.kind === 'block_read' && event.call_id === consumerCallId && event.block === block &&
    event.inputs?.some(input => input.node === writer.node && input.block === block && input.port === 'block'));
  if (!read) throw new Error(`no block_read from ${writer.node} to ${consumerCallId} for ${block}`);
  return { block, writer_call_id: writerCallId, writer_node: writer.node, consumer_call_id: consumerCallId,
    consumer_signature: signature, invocation_input_port: invocationEdge.port, block_read_node: read.node,
    ...(capturePort ? { capture_input_port: capturePort } : {}),
    ...(argumentMode === 'capture' ? {} : { argument_mode: argumentMode }),
    ...(writer.text_body_sha256 === undefined ? {} : { writer_body_sha256: writer.text_body_sha256 }),
    exact_runtime_writer_to_reader_link: true };
}
